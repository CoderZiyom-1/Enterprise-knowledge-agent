"""
Enterprise Knowledge Agent
--------------------------
Question -> plan search queries -> search the web (Tavily) -> store pages in Chroma
-> retrieve best chunks (trusted sources first) -> check if evidence is enough
(if not, search again) -> write a cited answer with conflicts and confidence.

Run:  python agent.py "How many employees does EdgeVerve have?"
"""

import os
import sys
import json
import uuid
from typing import TypedDict

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from trust import trust_tier, trust_score, find_citations, fix_confidence, remove_double_citations

load_dotenv()

MAX_ROUNDS = 2          
RESULTS_PER_QUERY = 5   
TOP_K_CHUNKS = 8        

llm = ChatGroq(model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"), temperature=0)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


class AgentState(TypedDict):
    question: str
    official_domain: str     
    collection: str          
    queries: list           
    searched: list           
    sources: dict            
    evidence: list           
    round: int
    enough: bool
    answer: dict


def ask_json(prompt):
    reply = llm.invoke(prompt)
    text = reply.content
    if isinstance(text, list):
        text = "".join(block.get("text", "") for block in text)
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

   
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception:
        print("Could not read JSON from the model. It said:")
        print(text)
        raise


def plan_queries(state):
    
    if state["round"] > 0:
        print("\n[plan] using follow-up queries:", state["queries"])
        return {}

    prompt = (
        "You are a research assistant. Write 3 different web search queries "
        "that together would answer this question.\n"
        "- Query 1: aimed at the OFFICIAL source (company website, annual report, press release).\n"
        "- Query 2: aimed at news or financial reporting.\n"
        "- Query 3: another angle, so we can compare facts across sources.\n"
        f"Question: {state['question']}\n"
        "Also give the official website domain of the main company in the question, "
        "like edgeverve.com or infosys.com (empty string if there is no company).\n"
        'Reply ONLY with JSON like {"queries": ["q1", "q2", "q3"], "official_domain": "example.com"}'
    )
    data = ask_json(prompt)
    domain = data.get("official_domain", "")
    print("\n[plan] queries:", data["queries"])
    print("[plan] official domain guess:", domain)
    return {"queries": data["queries"], "official_domain": domain}


def search_web(state):
    sources = dict(state["sources"])
    searched = list(state["searched"])

   
    if state["round"] == 0 and state["official_domain"] != "":
        print("[search] official site only:", state["official_domain"])
        try:
            response = tavily.search(
                query=state["question"],
                max_results=5,
                include_raw_content=True,
                include_domains=[state["official_domain"]],
            )
            for r in response["results"]:
                url = r["url"]
                text = r.get("raw_content")
                if not text:
                    text = r.get("content", "")
                text = text[:8000]
                if len(text) >= 100 and url not in sources:
                    sources[url] = {"title": r.get("title", url), "text": text}
        except Exception as e:
            print("   official search failed:", e)

    for q in state["queries"]:
        print("[search]", q)
        searched.append(q)
        try:
            response = tavily.search(
                query=q,
                max_results=RESULTS_PER_QUERY,
                include_raw_content=True,
            )
        except Exception as e:
            print("   search failed:", e)
            continue

        for r in response["results"]:
            url = r["url"]
            if url in sources:
                continue
            text = r.get("raw_content")
            if not text:
                text = r.get("content", "")
            text = text[:8000]   # do not store huge pages
            if len(text) < 100:
                continue
            sources[url] = {"title": r.get("title", url), "text": text}

    print("[search] total pages so far:", len(sources))
    return {"sources": sources, "searched": searched}


def sort_key(pair):
    
    return -pair[0]


def index_and_retrieve(state):
    db = Chroma(
        collection_name=state["collection"],
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    docs = []
    ids = []
    for url in state["sources"]:
        page = state["sources"][url]
        pieces = splitter.split_text(page["text"])
        number = 0
        for piece in pieces:
            doc = Document(page_content=piece, metadata={"url": url, "title": page["title"]})
            docs.append(doc)
            ids.append(url + "#" + str(number))
            number = number + 1

    db.add_documents(docs, ids=ids)

    
    found = db.similarity_search(state["question"], k=TOP_K_CHUNKS * 2)
    pairs = []
    for d in found:
        score = trust_score(d.metadata["url"], state["question"])
        pairs.append((score, d))
    pairs = sorted(pairs, key=sort_key)
    pairs = pairs[:TOP_K_CHUNKS]

    evidence = []
    for score, d in pairs:
        evidence.append({
            "text": d.page_content,
            "url": d.metadata["url"],
            "title": d.metadata["title"],
        })
    print("[retrieve] kept", len(evidence), "chunks")
    return {"evidence": evidence}


def check_enough(state):
    text = ""
    for e in state["evidence"]:
        tier = trust_tier(e["url"], state["question"])
        text = text + "- (" + tier + ") " + e["text"][:400] + "\n"

    prompt = (
        "You check research evidence.\n"
        f"Question: {state['question']}\n"
        f"Queries already searched: {state['searched']}\n"
        f"Evidence:\n{text}\n"
        "Is this enough to answer the question well? Evidence only from "
        "'aggregator' sites is NOT enough: we also need an official or news source. "
        "If something important is missing, write up to 2 NEW search queries "
        "(different from the ones already used).\n"
        'Reply ONLY with JSON: {"enough": true or false, "queries": ["..."]}'
    )
    data = ask_json(prompt)
    new_round = state["round"] + 1
    print(f"[check] round {new_round}, enough = {data['enough']}")
    return {
        "enough": data["enough"],
        "queries": data.get("queries", []),
        "round": new_round,
    }


def route_after_check(state):
    if state["enough"]:
        return "write_answer"
    if state["round"] >= MAX_ROUNDS:
        return "write_answer"
    if len(state["queries"]) == 0:
        return "write_answer"
    return "plan_queries"


def write_answer(state):
    question = state["question"]

    urls = []
    for e in state["evidence"]:
        if e["url"] not in urls:
            urls.append(e["url"])

    text = ""
    for e in state["evidence"]:
        number = urls.index(e["url"]) + 1
        tier = trust_tier(e["url"], question)
        text = text + f"[{number}] (trust: {tier}) {e['title']}\n{e['text']}\n\n"

    prompt = (
        "You are an enterprise knowledge assistant. Answer the question using ONLY "
        "the numbered sources below.\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n{text}\n"
        "Rules:\n"
        "1. Every sentence must end with a citation like [1] or [2][3]. "
        "No citation means do not write the sentence.\n"
        "2. Only state a fact (number, date, name, competitor) if a source says it. "
        "If the sources do not say it, write 'not found in the sources'.\n"
        "3. If sources disagree (different numbers, dates, facts), put it in 'conflicts'. "
        "Trust order when they disagree: official > reputable > other > aggregator. "
        "Aggregator sites copy each other, so several aggregators agreeing is weak proof. "
        "Mention the dates of the numbers if the sources give them.\n"
        "4. Keep the answer short and direct.\n"
        "5. confidence: 'High' = several independent sources agree. "
        "'Medium' = some gaps or small conflicts. 'Low' = few sources or weak evidence.\n\n"
        "Reply ONLY with JSON:\n"
        '{"answer": "...", "confidence": "High|Medium|Low", '
        '"confidence_reason": "...", '
        '"conflicts": [{"issue": "...", "resolution": "..."}]}'
    )
    data = ask_json(prompt)
    data["answer"] = remove_double_citations(data["answer"])


    numbers = find_citations(data["answer"], len(urls))
    if len(numbers) == 0:
        print("[answer] no citations found, asking again")
        retry_prompt = prompt + (
            "\n\nIMPORTANT: your last answer had NO citations. "
            "End every sentence with [1], [2] etc."
        )
        data = ask_json(retry_prompt)
        data["answer"] = remove_double_citations(data["answer"])
        numbers = find_citations(data["answer"], len(urls))

    
    cited_urls = []
    source_info = []
    for n in numbers:
        url = urls[n - 1]
        cited_urls.append(url)
        source_info.append({"n": n, "url": url, "trust": trust_tier(url, question)})

    
    conflicts = data.get("conflicts", [])
    if not isinstance(conflicts, list):
        conflicts = []
    final_level, note = fix_confidence(
        data.get("confidence", "Low"), cited_urls, question, len(conflicts) > 0
    )

    result = {
        "answer": data["answer"],
        "confidence": final_level,
        "confidence_reason": data.get("confidence_reason", ""),
        "confidence_note": note,
        "conflicts": conflicts,
        "sources": source_info,
    }
    return {"answer": result}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan_queries", plan_queries)
    graph.add_node("search_web", search_web)
    graph.add_node("index_and_retrieve", index_and_retrieve)
    graph.add_node("check_enough", check_enough)
    graph.add_node("write_answer", write_answer)

    graph.add_edge(START, "plan_queries")
    graph.add_edge("plan_queries", "search_web")
    graph.add_edge("search_web", "index_and_retrieve")
    graph.add_edge("index_and_retrieve", "check_enough")
    graph.add_conditional_edges(
        "check_enough",
        route_after_check,
        {"plan_queries": "plan_queries", "write_answer": "write_answer"},
    )
    graph.add_edge("write_answer", END)

    return graph.compile()


def ask(question):
    app = build_graph()
    start_state = {
        "question": question,
        "official_domain": "",
        "collection": "run_" + uuid.uuid4().hex[:8],
        "queries": [],
        "searched": [],
        "sources": {},
        "evidence": [],
        "round": 0,
        "enough": False,
        "answer": {},
    }
    final = app.invoke(start_state)
    return final["answer"]


def print_answer(result):
    print("\n" + "=" * 60)
    print("ANSWER\n")
    print(result["answer"])
    print("\nCONFIDENCE:", result["confidence"])
    if result["confidence_note"] != "":
        print("Note:", result["confidence_note"])
    print("Model's reason:", result["confidence_reason"])

    if len(result["conflicts"]) > 0:
        print("\nCONFLICTS FOUND")
        for c in result["conflicts"]:
            print("-", c["issue"])
            print("   resolution:", c["resolution"])

    print("\nSOURCES CITED")
    for s in result["sources"]:
        print(f"[{s['n']}] ({s['trust']}) {s['url']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question"')
        sys.exit(1)
    result = ask(sys.argv[1])
    print_answer(result)
