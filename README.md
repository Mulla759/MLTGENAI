# RAG-Powered Financial Analyst Chatbot

## Project Status: 🚧 Active Development 🚧

I'm currently building a high-impact, serverless solution that uses **Retrieval-Augmented Generation (RAG)** to instantly analyze and query hundreds of official SEC filings. This system is designed to replace time-consuming manual document research with a natural-language Q&A interface for investors.

## Features & Progress Tracker

This table tracks our progress on the core components of the RAG system and its deployment infrastructure.

| Feature Category | Status | Goal / Accomplishment | Technical Implementation |
| :--- | :--- | :--- | :--- |
| **Data Ingestion & RAG** | **[In Progress]** | Build the core RAG engine to ingest **200+ SEC filings** and create a structured, queryable knowledge base. | Implementing automated **CIK (Central Index Key)** lookup and **rate-limited ingestion** to comply with SEC API guidelines. |
| **Deployment & Architecture** | **[In Progress]** | Architect and deploy a highly available microservice solution on AWS. | Services will run on **AWS Lambda** (serverless compute), exposed via **API Gateway**, with documents and data stored in **AWS S3**. Aiming for **>99% uptime** via CI/CD pipelines. |
| **Performance & Validation** | **[To Do]** | Validate core investor use cases and measure impact: **cut manual research time from hours to seconds**. | Will include **S3 versioning** for data integrity and conducting pilot demos for user feedback. |

---

## Technical Stack

| Category | Components |
| :--- | :--- |
| **Generative AI / RAG** | RAG Architecture, Vector Database (**[To Be Determined]**), LLM (**[To Be Determined]**) |
| **Cloud / Infrastructure** | **AWS Lambda**, **AWS API Gateway**, **AWS S3** |
| **Development** | Python, Boto3/AWS SDK, CI/CD Pipelines |
| **Data Handling** | SEC API/EDGAR, CIK Lookup, Rate Limiting |

---

## In-Progress 🚧
