import os
import re
import pandas as pd
import numpy as np
import faiss
import requests
from tqdm import tqdm
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import streamlit as st
from googlesearch import search
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

######Piyush Waradkar###########

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  
FAISS_INDEX_PATH = "faiss_index.index"
EMBEDDINGS_PATH = "embeddings.npy"
DATA_CACHE_PATH = "data_cache.pkl"


def word_tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def sentence_tokenize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s]


class WikipediaConnector:
    def fetch(self, topic: str) -> List[Dict]:

        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={topic}&prop=extracts&exintro&explaintext"
        response = requests.get(wiki_url, timeout=10).json()
        page = next(iter(response['query']['pages'].values()))
        return [{
            "text": page['extract'],
            "source": "wikipedia",
            "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        }]


class GoogleSearchConnector:
    def fetch(self, query: str, num_results: int = 3) -> List[Dict]:
        results = list(
            search(query, num_results=num_results, advanced=True))
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self._fetch_url, url)
                       for url in results]
            return [f.result() for f in tqdm(futures, desc="Scraping URLs")]

    def _fetch_url(self, url):

        response = requests.get(url.url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return {
            "text": text[:5000],
            "source": "web",
            "url": url.url,
            "title": url.title
        }


##### Piyush Waradkar##########
class RestaurantRAG:
    def __init__(self, excel_path: str):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL, device=EMBEDDING_DEVICE)

        with st.spinner("Loading restaurant data..."):
            self.df = self._load_data(excel_path)
            self.restaurant_data = self._process_data()

        with st.spinner("Initializing search systems..."):
            self._initialize_retrieval()

        self.external_connectors = {
            "wikipedia": WikipediaConnector(),
            "google": GoogleSearchConnector()
        }

    @st.cache_data
    def _load_data(_self, excel_path: str) -> pd.DataFrame:
        if os.path.exists(DATA_CACHE_PATH):
            return pd.read_pickle(DATA_CACHE_PATH)
        df = pd.read_excel(excel_path)
        df.to_pickle(DATA_CACHE_PATH)
        return df

    def _process_data(self) -> List[Dict]:
        processed = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing rows"):
            text = (
                f"Restaurant: {row['restaurant_name']} | Category: {row['menu_category']} | "
                f"Item: {row['menu_item']} | Description: {row['menu_description']} | "
                f"Ingredients: {row['ingredient_name']} | Address: {row['address1']}, {row['city']} | "
                f"Price: {row['price']} | Rating: {row['rating']} | ID: {row['item_id']}"
            )
            processed.append({
                "text": text,
                "source": "internal",
                "id": str(row['item_id']),
                "price": row['price'],
                "category": row['menu_category']
            })
        return processed

    def _initialize_retrieval(self):
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAISS_INDEX_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            self.vector_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            texts = [d["text"] for d in self.restaurant_data]
            self.embeddings = self.model.encode(
                texts,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            np.save(EMBEDDINGS_PATH, self.embeddings)

            self.vector_index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.vector_index.add(self.embeddings.astype('float32'))
            faiss.write_index(self.vector_index, FAISS_INDEX_PATH)

        self.tokenized_docs = [word_tokenize(doc) for doc in tqdm(
            [d["text"] for d in self.restaurant_data],
            desc="Tokenizing docs"
        )]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode(
            [query], convert_to_tensor=True).cpu().numpy()
        _, vector_indices = self.vector_index.search(
            query_embedding.astype('float32'), top_k*2)

        tokenized_query = word_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]

        combined_indices = np.unique(
            np.concatenate([vector_indices[0], bm25_indices]))
        return [self.restaurant_data[i] for i in combined_indices[:top_k]]

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict], List[Dict]]:
        with st.spinner("Searching internal database..."):
            internal_results = self._hybrid_search(query, top_k)

        external_results = []
        with st.spinner("Gathering external context..."):
            for source in ['wikipedia', 'google']:
                external_data = self.external_connectors[source].fetch(
                    query)
                for doc in external_data:
                    if doc and doc['text']:
                        chunks = [doc['text'][i:i+500]
                                  for i in range(0, len(doc['text']), 500)]
                        external_results.extend([{
                            "text": chunk,
                            "source": doc['source'],
                            "url": doc.get('url', ''),
                            "title": doc.get('title', '')
                        } for chunk in chunks])

        return internal_results, external_results[:top_k]

    def generate_answer(self, query: str) -> Tuple[str, Dict]:
        internal, external = self.retrieve(query)
        prompt = self._build_prompt(query, internal, external)

        with st.spinner("Generating answer..."):
            answer = self._ask_llm(prompt)

        references = {
            "internal": [{"id": d["id"], "text": d["text"]} for d in internal],
            "external": [{
                "source": d["source"],
                "url": d["url"],
                "title": d["title"]
            } for d in external if d.get('url')]
        }

        return answer, references
#### Piyush Waradkar###
    def _build_prompt(self, query: str, internal: List[Dict], external: List[Dict]) -> str:
        context = []

        for doc in internal:
            context.append(f"【Menu Item {doc['id']}】\n{doc['text']}")

        for doc in external:
            src = f"{doc['source'].title()}: {doc.get('title', '')}"
            if doc.get('url'):
                src += f" ({doc['url']})"
            context.append(f"【{src}】\n{doc['text']}")

        return (
            "SYSTEM: You are a restaurant industry analyst. Use these verified sources:\n\n"
            "{context}\n\n"
            "USER QUERY: {query}\n\n"
            "ANSWER GUIDELINES:\n"
            "1. Cite sources using 【reference numbers】\n"
            "2. For prices, compare to category averages\n"
            "3. Separate facts from opinions\n"
            "4. Include both menu items and external context\n"
            "5. Acknowledge conflicting information\n\n"
            "ANSWER:"
        ).format(
            context='\n\n'.join(context),
            query=query
        )

    def _ask_llm(self, prompt: str) -> str:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': 'mistral', 'prompt': prompt, 'stream': False},
            timeout=60
        )
        return response.json()['response']


def main():
    st.set_page_config(page_title="Restaurant Analyst AI", layout="wide")

    if 'rag' not in st.session_state:
        with st.spinner("Initializing system (this will take a few minutes first time)..."):
            st.session_state.rag = RestaurantRAG(
                "Sample_Ingredients_File.xlsx")

    st.title("🔍 Restaurant Intelligent Assitant")


    query = st.text_input("Ask about restaurants, menus, or food trends:")
    if query:
        answer, references = st.session_state.rag.generate_answer(query)

        st.subheader("Expert Analysis")
        st.markdown(f"```\n{answer}\n```")

        with st.expander("Source References"):
            st.write("**Menu Items**")
            for item in references['internal']:
                st.caption(f"ID {item['id']}: {item['text'][:200]}...")

            if references['external']:
                st.write("**External Sources**")
                for src in references['external']:
                    st.markdown(f"- [{src['title']}]({src['url']})")

    


if __name__ == "__main__":
    main()

######Piyush Waradkar###########