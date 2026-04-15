# Medical Chatbot - Reverse Engineering Documentation

## Executive Summary

This document provides a comprehensive reverse engineering analysis of the Medical Chatbot project, a sophisticated Retrieval-Augmented Generation (RAG) system designed for medical Q&A applications. The system leverages advanced NLP techniques, multiple embedding models, and comprehensive evaluation methodologies to deliver accurate medical information retrieval and response generation.

---

## 1. Project Architecture Overview

### 1.1 System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend        │    │   Data Layer    │
│                 │    │                  │    │                 │
│ • Streamlit     │◄──►│ • LangChain      │◄──►│ • FAISS Vectors │
│ • Chat Interface│    │ • Groq LLM       │    │ • Medical PDFs  │
│ • Session Mgmt  │    │ • Embedding Mdl  │    │ • MedQuaD Dataset│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 1.2 Technology Stack
- **Frontend**: Streamlit 1.46.1
- **Backend**: LangChain 0.3.26
- **LLM**: Groq (Llama-3.1-8b-instant)
- **Vector Database**: FAISS 1.11.0
- **Embedding Models**: Multiple BioBERT variants
- **NLP Libraries**: NLTK, spaCy, Transformers
- **Evaluation**: scikit-learn, sentence-transformers

---

## 2. Core Application Analysis

### 2.1 Main Application (`medibot.py`)

**Purpose**: Primary chatbot interface with RAG capabilities

**Key Components**:
- **Vector Store Loading**: BioBERT embeddings with FAISS
- **LLM Integration**: Groq API for response generation
- **Chat History**: Session state management
- **Retrieval Chain**: LangChain RAG pipeline

**Configuration**:
```python
DB_FAISS_PATH = "vectorstore/db_faiss_sequential"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
RETRIEVAL_K = 3  # Top-k documents retrieved
```

**Data Flow**:
1. User input → Streamlit chat interface
2. Query embedding → FAISS similarity search
3. Retrieved documents → LangChain retrieval chain
4. Context + Query → Groq LLM for generation
5. Response → Streamlit display with history

### 2.2 Dataset-Based Chatbot (`usingDataset/medibot_data.py`)

**Purpose**: Alternative implementation using MedQuaD dataset

**Key Differences**:
- Uses MedQuaD FAISS index instead of PDF-based
- Enhanced metadata display (source, focus_area)
- Increased retrieval k=4 for better coverage

---

## 3. Data Processing Pipeline Analysis

### 3.1 Sequential Preprocessing (`hybrid_preprocessing.py`)

**Pipeline Stages**:
1. **Text Cleaning**: Lowercase, non-alphabetic removal
2. **Tokenization**: NLTK word tokenization
3. **Stopword Removal**: English stopwords filtering
4. **Lemmatization**: WordNet lemmatizer
5. **Stemming**: Porter stemmer
6. **POS-based Lemmatization**: spaCy POS tagging

**Performance Metrics**:
- Precision: Variable (0.0-1.0)
- Recall: Variable (0.0-0.11)
- F1 Score: Variable (0.0-0.2)

### 3.2 Dynamic Preprocessing (`create_dynamic.py`)

**Advanced Features**:
- **Pipeline Comparison**: Tests 7 different preprocessing strategies
- **Automated Selection**: Chooses best pipeline based on F1 score
- **Comprehensive Evaluation**: Precision, Recall, F1, Accuracy metrics

**Pipeline Options**:
- Raw (baseline)
- Clean (basic cleaning)
- Token (tokenization only)
- Stop (stopword removal)
- Lemma (lemmatization)
- Stem (stemming)
- POSLemma (POS-based lemmatization)

---

## 4. Vector Database Architecture

### 4.1 FAISS Implementation

**Index Types**:
- `db_faiss_sequential`: Sequential preprocessing
- `db_faiss_dynamic`: Dynamic preprocessing (best pipeline)
- `db_faiss_medical`: Medical-specific processing
- `medquad_faiss_sequential`: MedQuaD dataset-based

**Embedding Models Tested**:
```python
embedding_models = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "BioBERT": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "BGE-small": "BAAI/bge-small-en-v1.5"
}
```

### 4.2 Text Chunking Strategy

**Configuration**:
- Chunk Size: 700 characters
- Chunk Overlap: 100 characters
- Splitter: RecursiveCharacterTextSplitter

**Rationale**: Balances context preservation with retrieval granularity

---

## 5. Evaluation Framework Analysis

### 5.1 Embedding Model Comparison (`evaluate_embeddings.py`)

**Test Queries**: 28 comprehensive medical queries covering:
- Drug-related questions (4 queries)
- Disease symptoms (4 queries)
- Treatment/management (4 queries)
- Anatomy/physiology (4 queries)
- Pathology/diagnostics (4 queries)

**Evaluation Metrics**:
- **Precision**: Exact match accuracy
- **Recall**: Coverage of relevant terms
- **F1 Score**: Harmonic mean of precision/recall
- **Accuracy**: Overall correctness
- **Mean F1 Query**: Per-query average F1

**Semantic Matching**: Uses sentence-transformers with cosine similarity threshold of 0.45

### 5.2 Performance Results Summary

| Model | Precision | Recall | F1 Score | Accuracy | Mean F1 |
|-------|-----------|--------|----------|----------|---------|
| MiniLM | 1.0 | 0.111 | 0.200 | 0.111 | 0.160 |
| BioBERT | 1.0 | 0.042 | 0.080 | 0.042 | 0.065 |
| PubMedBERT | 1.0 | 0.014 | 0.027 | 0.014 | 0.025 |
| ClinicalBERT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MPNet | 1.0 | 0.083 | 0.154 | 0.083 | 0.120 |
| BGE-small | 1.0 | 0.056 | 0.105 | 0.056 | 0.090 |

**Key Insights**:
- MiniLM achieves best overall performance (F1: 0.200)
- ClinicalBERT failed completely (possible compatibility issue)
- All models show high precision but low recall
- Semantic matching threshold may need adjustment

### 5.3 MedQuaD Dataset Evaluation

**Dataset Characteristics**:
- Source: Medical Q&A pairs
- Size: ~22.8MB CSV file
- Columns: question, answer, source, focus_area

**Evaluation Strategy**:
- Train/Test split (90/10)
- Keyword extraction from answers
- Semantic similarity matching
- Per-query F1 scoring

---

## 6. Data Sources Analysis

### 6.1 Medical Documents
1. **The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf** (12.2MB)
   - Comprehensive medical encyclopedia
   - Primary knowledge source for PDF-based chatbot

2. **Additional Medical PDF** (61.1MB)
   - Supplementary medical document
   - Enhances knowledge base coverage

### 6.2 MedQuaD Dataset
- **Format**: CSV with Q&A pairs
- **Structure**: question, answer, source, focus_area
- **Usage**: Alternative knowledge source
- **Advantages**: Structured Q&A format, metadata-rich

---

## 7. System Performance Analysis

### 7.1 Strengths
1. **Comprehensive Evaluation**: Multiple metrics and models tested
2. **Flexible Architecture**: Modular design allows easy swapping
3. **Medical Specialization**: BioBERT and medical-specific embeddings
4. **Robust Preprocessing**: Multiple NLP pipelines compared
5. **Dataset Diversity**: Both PDF and structured Q&A sources

### 7.2 Performance Bottlenecks
1. **Low Recall**: All models show poor recall (0.0-0.111)
2. **ClinicalBERT Failure**: Complete model failure
3. **Semantic Threshold**: 0.45 threshold may be too restrictive
4. **Chunk Size**: 700 chars may be suboptimal for medical texts

### 7.3 Optimization Opportunities
1. **Threshold Tuning**: Adjust semantic similarity threshold
2. **Hybrid Retrieval**: Combine keyword + semantic search
3. **Larger Chunks**: Test 1000-1500 character chunks
4. **Ensemble Models**: Combine multiple embedding models
5. **Query Expansion**: Implement query augmentation

---

## 8. Security and Privacy Considerations

### 8.1 API Security
- Groq API key stored in environment variables
- No hardcoded credentials found
- Proper .env file usage

### 8.2 Data Privacy
- Medical data processing requires HIPAA compliance
- No patient-specific data identified in sources
- Public medical encyclopedias used as knowledge base

---

## 9. Deployment Architecture

### 9.1 Development Setup
- **Python Version**: 3.11+
- **Package Manager**: UV (uv.lock present)
- **Virtual Environment**: venv/.venv structure
- **Environment Variables**: .env configuration

### 9.2 Production Considerations
- **Scalability**: FAISS indices can be served independently
- **Caching**: Streamlit caching for vector store loading
- **Monitoring**: No logging/monitoring implementation found
- **Error Handling**: Basic try-catch in main application

---

## 10. Code Quality Assessment

### 10.1 Positive Aspects
- **Modular Design**: Clear separation of concerns
- **Configuration Management**: Centralized constants
- **Error Handling**: Basic exception handling
- **Documentation**: Inline comments and function docstrings

### 10.2 Areas for Improvement
- **Logging**: No structured logging implementation
- **Testing**: No unit tests found
- **Configuration**: Hardcoded model parameters
- **Validation**: Limited input validation

---

## 11. Recommendations

### 11.1 Immediate Improvements
1. **Fix ClinicalBERT**: Investigate and resolve model loading issues
2. **Threshold Optimization**: Experiment with semantic similarity thresholds
3. **Logging Implementation**: Add comprehensive logging
4. **Unit Testing**: Implement test suite for critical components

### 11.2 Medium-term Enhancements
1. **Hybrid Retrieval**: Implement keyword + semantic search
2. **Model Ensemble**: Combine multiple embedding models
3. **Query Expansion**: Add medical terminology expansion
4. **Performance Monitoring**: Implement metrics collection

### 11.3 Long-term Architecture
1. **Microservices**: Separate vector store service
2. **Caching Layer**: Redis for query/result caching
3. **Load Balancing**: Multiple model instances
4. **A/B Testing**: Framework for model comparison

---

## 12. Technical Debt Analysis

### 12.1 High Priority
- ClinicalBERT integration failure
- Low recall performance across all models
- Missing error recovery mechanisms

### 12.2 Medium Priority
- Hardcoded configuration values
- Limited input validation
- No performance monitoring

### 12.3 Low Priority
- Code documentation improvements
- Refactoring opportunities
- Enhanced user interface

---

## 13. Conclusion

The Medical Chatbot project demonstrates a sophisticated approach to building domain-specific RAG systems with comprehensive evaluation frameworks. While the architecture is sound and the evaluation methodology thorough, performance issues particularly with recall metrics and model compatibility need addressing.

The project's modular design and extensive evaluation provide a solid foundation for future enhancements. With the recommended optimizations, particularly in semantic threshold tuning and hybrid retrieval implementation, the system has significant potential for improved performance in medical Q&A applications.

**Overall Assessment**: Well-architected system with comprehensive evaluation but requiring performance optimization and model compatibility fixes.

---

*Documentation generated through reverse engineering analysis on March 31, 2026*
