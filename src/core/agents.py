import os
import logging
import sqlite3
import json
# Tipeado y validación
from typing import TypedDict, cast, Annotated
from pydantic import BaseModel, Field
# LangChain y LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
# Agentes y herramientas
from config import MODELO_AGENTE, LLM_TEMP_AGENTE, DATA_DIR
from tools.web_search import herramienta_web
from tools.wikipedia_search import herramienta_wiki
from tools.arxiv_search import herramienta_arxiv
from core.rag_base import consultar_mentor
from core.exceptions import RAGQueryError, QuotaExceededError, APIServiceUnavailableError

# 1. Configuración de Logging de Auditoría
logger = logging.getLogger("ai-mentor-hub.agents")
logger.setLevel(logging.INFO)
if not logger.handlers:
    consola = logging.StreamHandler()
    consola.setFormatter(logging.Formatter('\n%(levelname)s - %(message)s'))
    logger.addHandler(consola)

# 2. Definición del Estado
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]  # <-- AQUÍ ESTÁ LA MEMORIA
    pregunta: str
    razonamiento: str
    fuente_elegida: str
    query_busqueda: str
    contexto_recuperado: str
    borrador: str
    es_valido: bool
    intentos: int

# 3. Contratos de Pydantic (Descripciones en INGLÉS para ahorrar tokens)
class PlanAgente(BaseModel):
    razonamiento: str = Field(description="Step-by-step reasoning explaining why this source is the BEST for the question.")
    fuente_elegida: str = Field(description="Select ONLY ONE: 'RAG', 'WIKI', 'ARXIV', or 'WEB'.")
    query_optimizada: str = Field(description="Optimized search query extracted for the selected search tool.")

class Verificacion(BaseModel):
    es_valido: bool = Field(description="True if the draft accurately answers the question using the context. False if it hallucinates or is empty.")
    critica: str = Field(description="Feedback on what went wrong (if es_valido is False).")

# Variable global temporal para inyectar la Base de Datos Vectorial
_vector_db_global = None

def obtener_idioma_dominante() -> str:
    """Lee el idioma detectado desde los metadatos guardados por la ingesta."""
    ruta = os.path.join(DATA_DIR, "project_metadata.json")
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("idioma_rag", "Spanish")
        except Exception as e:
            logger.warning(f"[METADATA] No se pudo leer el archivo de idioma: {e}")
    return "Spanish"  # Default seguro

# --- FUNCIÓN AUXILIAR DE ERRORES ---
def _manejar_error_api(e: Exception, contexto_falla: str):
    """Filtra errores crudos de la API y lanza tus excepciones personalizadas."""
    error_str = str(e)
    logger.error(f"[{contexto_falla}] Error capturado: {error_str}")
    
    if "EXCEEDED_QUOTA" in error_str or "RESOURCE_EXHAUSTED" in error_str:
        raise QuotaExceededError("Se ha excedido la cuota de la API de Google.")
    elif "UNAVAILABLE" in error_str:
        raise APIServiceUnavailableError("El servicio de Gemini no está disponible temporalmente.")
    else:
        raise RAGQueryError(f"Error interno en {contexto_falla}: {error_str}")

# --- NODOS DEL GRAFO/Agentes ---

def planner_node(state: GraphState):
    """
    Analyzes the user's question and current conversation history to decide
    the best source of knowledge (RAG, Wikipedia, ArXiv, or Web).
    Uses Query Rewriting to ensure context continuity and adaptive language.
    
    Args:
        state (GraphState): Current graph state.
        
    Returns:
        dict: Updated state with reasoning, chosen source, and optimized query.
    """
    idioma_docs = obtener_idioma_dominante()
    logger.info(f"--- [PLANNER] Analizando: '{state['pregunta']}' | Idioma RAG: {idioma_docs} ---")
    
    try:
        # Extraemos contexto conversacional (últimos 6 mensajes) para Query Rewriting
        historial = state.get("messages", [])[:-1][-6:]
        historial_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in historial])

        llm = ChatGoogleGenerativeAI(model=MODELO_AGENTE, temperature=0.0) # Temp 0 para mayor precisión en routing
        llm_estructurado = llm.with_structured_output(PlanAgente)
        
        prompt = f"""You are the Master Routing Architect of a Bilingual AI-Mentor system. 
        
        TARGET SEARCH LANGUAGE: {idioma_docs}
        
        CONVERSATION HISTORY (Context for query rewriting):
        {historial_str if historial_str else "No prior history."}
        
        CURRENT USER QUESTION: {state['pregunta']}
        
        PREVIOUS ATTEMPT STATUS: {state.get("fuente_elegida", "First attempt")}
        FAILED CONTEXT: {state.get("contexto_recuperado", "None")}
        
        STRICT INSTRUCTIONS:
        1. RAG-FIRST MANDATE: If this is the 'First attempt' (intentos=0), YOU MUST CHOOSE 'RAG'. No exceptions.
        2. BILINGUAL QUERY: When choosing 'RAG', your 'query_optimizada' MUST be hybrid (Spanish and English). 
           Examples based on theme: 
           - (Cooking): "receta hamburguesas pavo - turkey burger recipe"
           - (Tech/Coding): "pasos para configurar python - python setup steps" or "gradiente descendiente - gradient descent"
           - (History): "revolución francesa - french revolution"
        3. FALLBACK: If the PREVIOUS ATTEMPT STATUS is 'RAG_FAILED', YOU MUST choose an alternative source (WEB, WIKI, or ARXIV).
        4. AGGRESSIVE QUERY REWRITING: 
           - If the user asks for a part/follow-up (e.g., "ingredients", "steps", "explain more"), YOU MUST extract the EXACT FULL TOPIC NAME from the CONVERSATION HISTORY and prepend it to the query.
           - If the user asks for a complete process, guide, or entity (e.g., "how to do X", "the recipe for Y", "guide for Z"), ensure your query includes broad terms to capture the ENTIRE entity (e.g., instead of just 'Python setup', use 'Python setup prerequisites configuration steps').
        
        ROUTING RULES:
        - "RAG": MANDATORY for first attempt. Primary for local PDFs.
        - "WIKI": Formal definitions or history (Only if RAG failed).
        - "ARXIV": Scientific research (Only if RAG failed).
        - "WEB": Current news or if RAG/others failed.
        """
        
        plan = cast(PlanAgente, llm_estructurado.invoke(prompt))
        
        # --- CODE-LEVEL RAG-FIRST ENFORCEMENT ---
        final_source = plan.fuente_elegida
        if state.get("intentos", 0) == 0:
            final_source = "RAG"
            logger.info(f"[PLANNER] Forzando RAG en el primer intento.")
        
        logger.info(f"[PLANNER] Decisión: {final_source} | Búsqueda: '{plan.query_optimizada}'")
        
        return {
            "razonamiento": plan.razonamiento,
            "fuente_elegida": final_source,
            "query_busqueda": plan.query_optimizada,
            "intentos": state.get("intentos", 0)
        }
    except Exception as e:
        _manejar_error_api(e, "PLANNER")

def retriever_node(state: GraphState):
    """
    Executes the search and applies Deterministic Rerank (sorting by page).
    If no information is found in RAG, triggers a fallback flag.
    
    Args:
        state (GraphState): Current graph state.
        
    Returns:
        dict: Updated state with the retrieved context.
    """
    fuente = state["fuente_elegida"]
    query = state["query_busqueda"]
    logger.info(f"--- [RETRIEVER] Ejecutando búsqueda en {fuente}... ---")
    
    contexto = ""
    try:
        if fuente == "RAG" and _vector_db_global is not None:
            from config import RETRIEVAL_K
            from langchain_core.documents import Document
            
            # 1. Búsqueda inicial (Menor ruido, usamos RETRIEVAL_K que ahora es 6)
            docs = _vector_db_global.similarity_search(query, k=RETRIEVAL_K)
            
            # --- DETECCIÓN DE VACÍO (FALLBACK) ---
            if not docs or len(docs) == 0:
                logger.warning(f"[RETRIEVER] No se encontró información relevante en RAG para: '{query}'")
                return {"contexto_recuperado": "FALLBACK_TRIGGERED", "fuente_elegida": "RAG_FAILED"}

            # --- NEIGHBOR EXPANSION (Fix #2) ---
            # 2. Identificar páginas base y sus vecinos
            paginas_objetivo = set()
            for doc in docs:
                src = doc.metadata.get('source')
                pag = doc.metadata.get('page')
                if src is not None and pag is not None:
                    paginas_objetivo.add((src, pag))
                    paginas_objetivo.add((src, pag - 1)) # Página anterior
                    paginas_objetivo.add((src, pag + 1)) # Página siguiente
                    
            # 3. Extraer todos los fragmentos de las páginas objetivo
            docs_expandidos = []
            try:
                for doc in _vector_db_global.doc_index._docs:
                    src = doc.metadata.get('source')
                    pag = doc.metadata.get('page')
                    if (src, pag) in paginas_objetivo:
                        docs_expandidos.append(Document(page_content=doc.text, metadata=doc.metadata))
            except Exception as e:
                logger.warning(f"[RETRIEVER] Fallo en Neighbor Expansion, usando docs originales: {e}")
                docs_expandidos = docs

            # 4. Ordenar determinísticamente
            docs_ordenados = sorted(docs_expandidos, key=lambda x: (x.metadata.get('source', ''), x.metadata.get('page', 0)))
            
            # 5. Filtrar y ensamblar
            contexto_parts = []
            textos_vistos = set()
            for doc in docs_ordenados:
                if doc.page_content not in textos_vistos:
                    textos_vistos.add(doc.page_content)
                    pag = doc.metadata.get('page', 'S/N')
                    src = os.path.basename(doc.metadata.get('source', 'Biblioteca'))
                    contexto_parts.append(f"--- Fuente: {src} | Pág: {pag} ---\n{doc.page_content}")
            
            contexto = "\n\n".join(contexto_parts)
            logger.info(f"[RAG] Expansión exitosa: De {len(docs)} a {len(contexto_parts)} fragmentos ordenados.")
            
        elif fuente == "WIKI":
            contexto = herramienta_wiki.invoke(query)
        elif fuente == "ARXIV":
            contexto = herramienta_arxiv.invoke(query)
        else:
            contexto = herramienta_web.invoke(query)
            
        # Si las herramientas externas devuelven algo muy corto o vacío
        if not contexto or len(contexto.strip()) < 50:
             return {"contexto_recuperado": "FALLBACK_TRIGGERED", "fuente_elegida": f"{fuente}_FAILED"}
            
    except Exception as e:
        if isinstance(e, (QuotaExceededError, APIServiceUnavailableError)):
            raise e
        logger.warning(f"[RETRIEVER] Falla en la herramienta {fuente}: {str(e)}")
        return {"contexto_recuperado": "FALLBACK_TRIGGERED", "fuente_elegida": f"{fuente}_ERROR"}
        
    return {"contexto_recuperado": contexto}
def executor_node(state: GraphState):
    """
    Generates a final answer in Spanish, translating if the context is in English.
    
    Args:
        state (GraphState): Current graph state.
        
    Returns:
        dict: Updated state with the draft answer and updated message history.
    """
    logger.info("--- [EXECUTOR] Redactando respuesta final bilingüe... ---")
    
    try:
        # Extraemos contexto conversacional (últimos 4 mensajes) para dar memoria al Executor
        historial = state.get("messages", [])[:-1][-4:]
        historial_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in historial])

        llm = ChatGoogleGenerativeAI(model=MODELO_AGENTE, temperature=LLM_TEMP_AGENTE)
        
        prompt = f"""You are a High-Fidelity Data Extractor and AI Mentor. Your goal is to provide information with 100% fidelity based on the context and the conversation history.
        
        CONVERSATION HISTORY (Use this to understand follow-up questions like '18' or 'give me the ingredients'):
        {historial_str if historial_str else "No prior history."}
        
        CURRENT QUESTION: {state['pregunta']}
        
        CONTEXT SOURCE: {state['fuente_elegida']}
        CONTEXT: {state['contexto_recuperado']}
        
        STRICT RULES:
        1. CONTEXT AWARENESS: If the CURRENT QUESTION is a short reply (e.g., "18", "yes", "the first one"), use the CONVERSATION HISTORY to understand what the user is referring to before extracting the data from the CONTEXT.
        2. TOPIC CONTINUITY: If the user asks a follow-up question, use the CONVERSATION HISTORY to identify the topic. Scan the CONTEXT specifically for the recipe/topic discussed in the previous message. However, if the exact topic is not found (e.g., slight name variation), search the context for the closest matching subject before declaring a failure.
        3. ANTI-INDEX RULE: IGNORE table of contents, indexes, or generic lists of recipes in the CONTEXT. Focus ONLY on the actual content/pages that contain the detailed information requested.
        4. VERBATIM EXTRACTION: When asked for procedures, steps, instructions, or lists of any kind, you MUST act as a strict copy-paste tool for the content. Provide the EXACT verbatim text from the document, without summarizing or omitting details.
        5. FORMATTING: You MUST restore logical formatting. Even when extracting verbatim content, present lists (like ingredients) as bulleted lists (`- item`) and procedures/steps as numbered or bulleted lists with line breaks. NEVER output a solid wall of text for structured data.
        6. COMPLETE ENTITIES: If the user asks for a specific topic, concept, or process (e.g., "the recipe for X", "how to install Y"), you MUST extract ALL sections related to it found in the context (e.g., BOTH ingredients AND preparation, or BOTH prerequisites AND steps). Do not stop halfway through the provided context.
        7. FALLBACK SIGNAL: If the required information is definitely NOT in the provided CONTEXT, you MUST output exactly the word "NOT_FOUND_IN_CONTEXT". Do not apologize, do not explain why, and do not suggest alternatives. Just output that one phrase.
        8. TRANSPARENCY: If the source is 'WEB', 'WIKI', or 'ARXIV' as a result of a RAG failure, start by saying: "No encontré esto en tus documentos, pero busqué en [FUENTE] y esto es lo que encontré:".
        9. ACCURACY: DO NOT invent facts. Use only the provided context.
        10. FINAL ANSWER MUST BE ENTIRELY IN SPANISH.
        """
        
        respuesta = llm.invoke(prompt)
        
        raw_content = respuesta.content
        if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict):
            texto_final = raw_content[0].get('text', str(raw_content)).strip()
        else:
            texto_final = str(raw_content).strip()
        
        # Si el modelo indica que no encontró nada, forzamos el estado de fallback
        if "NOT_FOUND_IN_CONTEXT" in texto_final:
            logger.warning("[EXECUTOR] Información no encontrada en el contexto actual. Activando señal de Fallback.")
            return {
                "borrador": "NOT_FOUND_IN_CONTEXT",
                "contexto_recuperado": "FALLBACK_TRIGGERED"
            }
        
        logger.info("[EXECUTOR] Borrador finalizado.")
        
        # Guardamos la respuesta del LLM en el historial oficial de la memoria
        return {
            "borrador": texto_final,
            "messages": [AIMessage(content=texto_final)]
        }
    except Exception as e:
        _manejar_error_api(e, "EXECUTOR")

def verifier_node(state: GraphState):
    """
    Audits the generated response to ensure it accurately reflects the retrieved context
    and doesn't contain hallucinations.
    
    Args:
        state (GraphState): Current graph state.
        
    Returns:
        dict: Updated state with the validation flag and incremented attempt counter.
    """
    logger.info("--- [VERIFIER] Auditando borrador contra alucinaciones... ---")
    
    intentos = state["intentos"] + 1
    try:
        llm = ChatGoogleGenerativeAI(model=MODELO_AGENTE, temperature=0.0)
        llm_estructurado = llm.with_structured_output(Verificacion)
        
        prompt = f"""Audit this response. Verify if the draft accurately answers the question based on the context.
        Question: {state['pregunta']}
        Context: {state['contexto_recuperado']}
        Draft to evaluate: {state['borrador']}
        
        RULES:
        1. Be forgiving of minor name variations (e.g., if the context says "Salteado de pollo" and the draft says "Salteado de pollo y verduras", that is valid).
        2. Reject (es_valido=False) ONLY if the draft completely invents facts, ingredients, or steps that are definitely not in the context.
        """
        
        verificacion = cast(Verificacion, llm_estructurado.invoke(prompt))
        
        logger.info(f"[VERIFIER] ¿Aprobado?: {verificacion.es_valido} | Intento: {intentos}")
        if not verificacion.es_valido:
            logger.warning(f"[VERIFIER] Crítica: {verificacion.critica}")
            
        return {"es_valido": verificacion.es_valido, "intentos": intentos}
    except Exception as e:
        # Si el Verifier falla por un error temporal de parseo, aprobamos para evitar loops infinitos
        logger.error(f"[VERIFIER] Error al auditar (Intento {intentos}): {e}")
        return {"es_valido": True, "intentos": intentos}

def decidir_post_retrieval(state: GraphState):
    """
    Decides whether to proceed to execution or loop back to planning if a fallback is triggered.
    """
    if state["contexto_recuperado"] == "FALLBACK_TRIGGERED":
        if state["intentos"] >= 2:
            logger.warning("--- [FLUJO] Fallback activado pero máximo de intentos alcanzado. Pasando a Executor. ---")
            return "executor"
        logger.warning(f"--- [FLUJO] Fallback activado ({state['fuente_elegida']}). Regresando al Planner. ---")
        return "planner"
    return "executor"

def decidir_flujo(state: GraphState):
    """
    Determines whether to finish the execution or return to the planner for another attempt.
    
    Args:
        state (GraphState): Current graph state.
        
    Returns:
        str: Next node to execute (END or 'planner').
    """
    if state["es_valido"] or state["intentos"] >= 3: # Aumentado a 3 para permitir fallback + refinamiento
        logger.info("--- [FLUJO] Aprobado. Terminando grafo. ---")
        return END
    else:
        logger.warning("--- [FLUJO] Rechazado. Regresando al Planner para reformular. ---")
        return "planner"

# --- CONSTRUCCIÓN DEL GRAFO ---
try:
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    
    # Cambio a arista condicional para manejar el Fallback
    workflow.add_conditional_edges(
        "retriever", 
        decidir_post_retrieval, 
        {"planner": "planner", "executor": "executor"}
    )
    
    workflow.add_edge("executor", "verifier")

    workflow.add_conditional_edges("verifier", decidir_flujo, {END: END, "planner": "planner"})

    # Configuración de SQLite para Checkpointing (Memoria Persistente)
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(os.path.join(DATA_DIR, "agent_memory.db"), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    app_grafo = workflow.compile(checkpointer=checkpointer)
    logger.info("✅ Grafo Multiagente compilado exitosamente con SQLite.")
except Exception as e:
    logger.critical(f"❌ Error crítico al compilar el Grafo LangGraph: {e}")
    raise e

def ejecutar_grafo_multiagente(pregunta: str, vector_db, thread_id: str = "1") -> str:
    """
    Main entry point for Streamlit to execute the multi-agent orchestration graph.
    
    Args:
        pregunta (str): User query.
        vector_db: Vector database instance for RAG retrieval.
        thread_id (str): Unique session identifier for memory persistence.
        
    Returns:
        str: The final generated response.
        
    Raises:
        QuotaExceededError: API limit reached.
        APIServiceUnavailableError: Service down.
        RAGQueryError: Unexpected orchestration failure.
    """
    global _vector_db_global
    _vector_db_global = vector_db
    
    config = {"configurable": {"thread_id": thread_id}}
    
    estado_inicial = {
        "pregunta": pregunta,
        "messages": [HumanMessage(content=pregunta)], # Registramos la pregunta en SQLite
        "intentos": 0
    }
    
    try:
        resultado = app_grafo.invoke(estado_inicial, config=config)
        return resultado.get("borrador", "Lo siento, no pude generar una respuesta.")
    except Exception as e:
        if isinstance(e, (QuotaExceededError, APIServiceUnavailableError, RAGQueryError)):
            raise e
        raise RAGQueryError(f"Error inesperado en la orquestación: {str(e)}")    
