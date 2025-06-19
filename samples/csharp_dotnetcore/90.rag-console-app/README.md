# RAG Console Application

This sample demonstrates the Retrieval Augmented Generation (RAG) pattern using Azure services. The application uploads documents to Azure Blob Storage, indexes them in Azure AI Search, and uses Azure OpenAI to generate contextual answers to user questions.

## Features

1. **Document Upload**: Upload text files to Azure Blob Storage
2. **Document Indexing**: Index document content in Azure AI Search for fast retrieval
3. **Interactive Console**: Ask questions through a simple console interface
4. **Context Retrieval**: Retrieve relevant document chunks from Azure AI Search
5. **AI-Generated Answers**: Use Azure OpenAI to generate answers based on retrieved context
6. **Configuration Management**: Easy configuration for all Azure services

## Prerequisites

- Azure subscription
- Azure Blob Storage account
- Azure AI Search service
- Azure OpenAI service with a deployed model (e.g., GPT-3.5-turbo or GPT-4)
- .NET 8.0 SDK

## Azure Service Setup

### 1. Azure Blob Storage
1. Create a Storage Account in Azure
2. Copy the connection string from the Access Keys section

### 2. Azure AI Search
1. Create an Azure AI Search service
2. Note the service URL and admin key from the Keys section

### 3. Azure OpenAI
1. Create an Azure OpenAI resource
2. Deploy a chat model (e.g., gpt-35-turbo or gpt-4)
3. Note the endpoint URL, API key, and deployment name

## Configuration

Update the `appsettings.json` file with your Azure service details:

```json
{
  "AzureBlobStorage": {
    "ConnectionString": "your-blob-storage-connection-string",
    "ContainerName": "documents"
  },
  "AzureAISearch": {
    "ServiceEndpoint": "https://your-search-service.search.windows.net",
    "ApiKey": "your-search-admin-key",
    "IndexName": "rag-documents"
  },
  "AzureOpenAI": {
    "Endpoint": "https://your-openai-resource.openai.azure.com/",
    "ApiKey": "your-openai-api-key",
    "DeploymentName": "your-model-deployment-name"
  }
}
```

## Running the Application

1. Navigate to the project directory:
   ```bash
   cd samples/csharp_dotnetcore/90.rag-console-app
   ```

2. Build the application:
   ```bash
   dotnet build
   ```

3. Run the application:
   ```bash
   dotnet run
   ```

## Usage

The application will:
1. Initialize the Azure AI Search index
2. Upload and index the sample document (`sample-document.txt`) if it exists
3. Start an interactive console session

### Commands

- **Ask questions**: Simply type your question and press Enter
- **Upload documents**: Type `upload <filepath>` to upload and index a new document
- **Exit**: Type `exit` to quit the application

### Example Session

```
=== RAG (Retrieval Augmented Generation) Console Application ===

Initializing RAG system...
Search index 'rag-documents' created successfully.
RAG system initialized successfully.
Sample document found. Uploading and indexing...
Document 'sample-document.txt' successfully uploaded and indexed.

RAG system is ready! You can now ask questions.
Available commands:
- Ask any question to get an AI-generated answer based on indexed documents
- Type 'upload <filepath>' to upload and index a new document
- Type 'exit' to quit

You: What is Azure Blob Storage?
Searching for relevant documents for question: 'What is Azure Blob Storage?'
Found 1 relevant document(s). Generating answer...
Assistant: Azure Blob Storage is a service for storing large amounts of unstructured object data, such as text or binary data, that can be accessed from anywhere in the world via HTTP or HTTPS. You can use Blob storage to expose data publicly to the world, or to store application data privately.

You: What is the RAG pattern?
Searching for relevant documents for question: 'What is the RAG pattern?'
Found 1 relevant document(s). Generating answer...
Assistant: The Retrieval Augmented Generation (RAG) pattern combines the power of large language models with the ability to retrieve relevant information from external knowledge sources. This approach allows for more accurate and contextually relevant responses by grounding the model's output in factual information retrieved from a knowledge base.

You: exit
Goodbye!
```

## Architecture

The application follows a clean architecture pattern with the following components:

- **Configuration**: Strongly-typed configuration classes for Azure services
- **Models**: Data models for search documents
- **Services**:
  - `BlobStorageService`: Handles document upload and retrieval from Azure Blob Storage
  - `AISearchService`: Manages document indexing and search operations
  - `OpenAIService`: Generates AI responses using Azure OpenAI
  - `RagService`: Orchestrates the RAG workflow

## Sample Document

The application includes a sample document (`sample-document.txt`) that contains information about Azure services and the RAG pattern. This document will be automatically uploaded and indexed when you first run the application.

## Error Handling

The application includes comprehensive error handling for:
- Missing configuration values
- Azure service connection issues
- File upload errors
- Search and indexing failures
- AI response generation errors

## Further Reading

- [Azure Blob Storage Documentation](https://docs.microsoft.com/azure/storage/blobs/)
- [Azure AI Search Documentation](https://docs.microsoft.com/azure/search/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401)