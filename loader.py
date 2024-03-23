import os
import requests
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
import streamlit as st
from streamlit.logger import get_logger
from chains import load_embedding_model
from utils import create_constraints, create_vector_index
from PIL import Image
import praw
from trafilatura import fetch_url, extract
import traceback
import sys

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

create_constraints(neo4j_graph)
create_vector_index(neo4j_graph, dimension)
reddit = praw.Reddit(
    client_id=os.getenv("PRAW_ID"),
    client_secret="yYuIt0aDZe2OSB54Z0zzBey-x3ulLQ",
    user_agent="Mozilla/5.0 (X11; CrOS x86_64 15633.69.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.212 Safari/537.36",
)

def redditScraper(subReddit, amountOfPosts=100, topOfWhat='month'):
    listOfPosts = []
    for submission in reddit.subreddit(subReddit).top(topOfWhat, limit=amountOfPosts):
        
        post_obj = {}
        post_obj["url"] = submission.url
        post_obj["id"] = submission.id
        post_obj["author"] = submission.author.name
        post_obj["domain"] = submission.domain
        post_obj["created"] = submission.created
        post_obj["created_utc"] = submission.created_utc
        post_obj["title"] = submission.title
        post_obj["selftext"] = submission.selftext
        downloaded = fetch_url(submission.url)
        post_obj["content"] = extract(downloaded)
        post_obj["score"] = submission.upvote_ratio
        submission.comments.replace_more(limit=0)
        post_obj["comments"] = [{'author':comment.author.name if comment.author else "unknown", 
                                'text': comment.body, 
                                'ups': comment.ups } for comment in submission.comments if comment.ups > 100
                                ]
                                                                
        listOfPosts.append(post_obj)

    print ("Grabbed " + str(len(listOfPosts)) + " posts from r/" + subReddit)
    return listOfPosts

def load_sub_data(subreddit: str = "news", page: int = 1) -> None:
    data = redditScraper(subreddit)
    insert_reddit_data(data)

def insert_reddit_data(data) -> None:
    for q in data:
        question_text = q["title"] + "\n" + q["selftext"]
        q["embedding"] = embeddings.embed_query(question_text)
        for a in q["comments"]:
            a["embedding"] = embeddings.embed_query(
                question_text + "\n" + a['author'] + " commented " + a['text']
            )
    import_query = """
    UNWIND $data AS post
    MERGE (submission:Submission {id: post.id}) 
    ON CREATE SET submission.title = post.title, submission.url = post.url,
                submission.author = post.author, submission.domain = post.domain,
                submission.text = post.selftext, submission.embedding = post.embedding,
                submission.score = post.score
    FOREACH (comment_data IN post.comments |
        MERGE (comment:Comment {body: comment_data.text})
        ON CREATE SET comment.author = comment_data.author, 
                    comment.embedding = comment_data.embedding ,
                    comment.upvotes = comment_data.ups
        MERGE (submission)<-[:HAS_COMMENT]-(comment)
    )
    """
    # print(data)
    neo4j_graph.query(import_query, {"data": data})


def load_high_score_so_data() -> None:
    parameters = (
        f"?fromdate=1664150400&order=desc&sort=votes&site=stackoverflow&"
        "filter=!.DK56VBPooplF.)bWW5iOX32Fh1lcCkw1b_Y6Zkb7YD8.ZMhrR5.FRRsR6Z1uK8*Z5wPaONvyII"
    )
    data = requests.get(so_api_base_url + parameters).json()
    insert_so_data(data)

def insert_so_data(data: dict) -> None:
    # Calculate embedding values for questions and answers
    for q in data["items"]:
        question_text = q["title"] + "\n" + q["body_markdown"]
        q["embedding"] = embeddings.embed_query(question_text)
        for a in q["content"].split("\n"):
            a["embedding"] = embeddings.embed_query(
                question_text + "\n" + a["body_markdown"]
            )

    # Cypher, the query language of Neo4j, is used to import the data
    # https://neo4j.com/docs/getting-started/cypher-intro/
    # https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise/
    import_query = """
    UNWIND $data AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
        question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
        question.body = q.body_markdown, question.embedding = q.embedding
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = datetime({epochSeconds:a.creation_date}),
            answer.body = a.body_markdown,
            answer.embedding = a.embedding
        MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
        ON CREATE SET answerer.display_name = a.owner.display_name,
                      answerer.reputation= a.owner.reputation
        MERGE (answer)<-[:PROVIDED]-(answerer)
    )
    WITH * WHERE NOT q.owner.user_id IS NULL
    MERGE (owner:User {id:q.owner.user_id})
    ON CREATE SET owner.display_name = q.owner.display_name,
                  owner.reputation = q.owner.reputation
    MERGE (owner)-[:ASKED]->(question)
    """
    neo4j_graph.query(import_query, {"data": data["items"]})


# Streamlit
def get_tag():
    col1, col2 = st.columns(2)
    with col1:
        input_text = st.text_input(
            "Which subreddits do you want to import?", value="neo4j"
        )   
    with col2:
        num_items = st.number_input(
            "Number of items", step=1, min_value=3
        )
    return input_text,num_items

def render_page():
    datamodel_image = Image.open("./images/datamodel.png")
    st.header("Subreddit Loader")
    st.subheader("Choose subreddits,subreddit2 to load into Neo4j")
    st.caption("Go to http://localhost:7474/ to explore the graph.")

    user_input,num_items = get_tag()
    # num_pages, start_page = get_pages()

    if st.button("Import", type="primary"):
        with st.spinner("Loading... This might take a minute or two."):
            try:
                # for page in range(1, num_pages + 1):
                load_sub_data(user_input, num_items)
                st.success("Import successful", icon="✅")
                st.caption("Data model")
                st.image(datamodel_image)
                st.caption("Go to http://localhost:7474/ to interact with the database")
            except Exception as e:
                print(e)
                st.error(f"Error: {traceback.format_exc()}", icon="🚨")
    with st.expander("Highly Stack overflow questions rather than reddit?"):
        if st.button("Import highly ranked questions"):
            with st.spinner("Loading... This might take a minute or two."):
                try:
                    load_high_score_so_data()
                    st.success("Import successful", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}", icon="🚨")


render_page()
