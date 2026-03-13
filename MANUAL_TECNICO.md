---

### Manual Técnico de Arquitectura (Para tu entrega)

Puedes guardar esto como `MANUAL_TECNICO.md` o incluirlo como sección en tu reporte de la maestría.

```markdown
# Manual Técnico y Decisiones de Arquitectura (ADR)

Este documento detalla el flujo de datos y las decisiones de ingeniería tomadas para la construcción del AI-Mentor Hub.

## 1. Flujo de Datos (Pipeline RAG)

1.  **Extracción y Fragmentación (`src/ingestion.py`):** Los documentos en `data/` se procesan con `PyPDFLoader` y se dividen con `RecursiveCharacterTextSplitter` (chunks de 1000 caracteres, overlap de 150) para mantener el contexto semántico.
2.  **Vectorización Local:** Se utiliza `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) operando puramente sobre la CPU.
3.  **Indexación y Persistencia:** Los vectores se cargan en `DocArrayInMemorySearch`. Para evitar errores de memoria o incompatibilidades de librerías de C++ (como FAISS o ChromaDB), la persistencia del estado se realiza exportando los metadatos y el texto a un archivo `processed_docs.json`.
4.  **Recuperación (Retrieval):** Al recibir un prompt, el sistema busca los 4 fragmentos (`k=4`) con mayor similitud de coseno.
5.  **Generación Estructurada (`src/rag_base.py`):** El contexto se inyecta en un prompt diseñado para el modelo `gemini-2.5-flash`. Se utiliza `GenerateContentConfig` para obligar al modelo a devolver un objeto que cumpla con el esquema definido en `src/schemas.py`.

## 2. Registro de Decisiones Arquitectónicas (ADR)

- **ADR 001: Sustitución de Embeddings de API por Modelos Locales.**
  - _Contexto:_ Las cuotas gratuitas de APIs en la nube limitan severamente el proceso de ingesta masiva (Rate Limiting de 15-100 RPM).
  - _Decisión:_ Implementar `sentence-transformers` ejecutándose localmente.
  - _Consecuencia:_ Ingesta ilimitada y sin costo, a cambio de una carga inicial en CPU.
- **ADR 002: Control estricto de dependencias de C-API (`numpy<2`).**
  - _Contexto:_ Las versiones modernas de PyTorch y librerías científicas requieren instrucciones de procesador (AVX2) no disponibles en el hardware de desarrollo (Intel Core de 3ra generación). La actualización a NumPy 2.0 rompe la compatibilidad binaria con versiones anteriores de PyTorch.
  - _Decisión:_ Hacer un _downgrade_ táctico anclando `numpy<2`, `transformers<4.45` y `pyarrow<15`.
  - _Consecuencia:_ Estabilidad total del sistema sin comprometer las funcionalidades del RAG.
- **ADR 003: Persistencia en JSON vs Pickle.**
  - _Contexto:_ Serializar objetos complejos de LangChain con `pickle` genera errores de "Attribute lookup" al recargar el programa.
  - _Decisión:_ Extraer los datos crudos (`page_content` y `metadata`) y guardarlos en JSON estándar.
  - _Consecuencia:_ Cargas de entorno seguras, independientes de la versión de Python y fáciles de auditar.
```
