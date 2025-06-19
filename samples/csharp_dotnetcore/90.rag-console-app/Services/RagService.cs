// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using RagConsoleApp.Models;

namespace RagConsoleApp.Services
{
    public interface IRagService
    {
        Task InitializeAsync();
        Task<string> UploadAndIndexDocumentAsync(string filePath);
        Task<string> AnswerQuestionAsync(string question);
    }

    public class RagService : IRagService
    {
        private readonly IBlobStorageService _blobStorageService;
        private readonly IAISearchService _searchService;
        private readonly IOpenAIService _openAIService;

        public RagService(
            IBlobStorageService blobStorageService,
            IAISearchService searchService,
            IOpenAIService openAIService)
        {
            _blobStorageService = blobStorageService;
            _searchService = searchService;
            _openAIService = openAIService;
        }

        public async Task InitializeAsync()
        {
            Console.WriteLine("Initializing RAG system...");
            await _searchService.InitializeIndexAsync();
            Console.WriteLine("RAG system initialized successfully.");
        }

        public async Task<string> UploadAndIndexDocumentAsync(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileName = Path.GetFileName(filePath);
            var content = await File.ReadAllTextAsync(filePath);
            var title = Path.GetFileNameWithoutExtension(fileName);

            Console.WriteLine($"Uploading document '{fileName}' to blob storage...");
            var blobUri = await _blobStorageService.UploadDocumentAsync(fileName, content);

            Console.WriteLine($"Indexing document '{fileName}' in AI Search...");
            var document = new DocumentModel
            {
                Id = Guid.NewGuid().ToString(),
                Title = title,
                Content = content,
                FileName = fileName,
                UploadedAt = DateTime.UtcNow,
                BlobUri = blobUri
            };

            await _searchService.IndexDocumentAsync(document);
            
            return $"Document '{fileName}' successfully uploaded and indexed.";
        }

        public async Task<string> AnswerQuestionAsync(string question)
        {
            Console.WriteLine($"Searching for relevant documents for question: '{question}'");
            
            var relevantDocs = await _searchService.SearchDocumentsAsync(question);
            
            if (!relevantDocs.Any())
            {
                return "I couldn't find any relevant documents to answer your question. Please make sure documents have been uploaded and indexed.";
            }

            Console.WriteLine($"Found {relevantDocs.Count} relevant document(s). Generating answer...");
            
            var answer = await _openAIService.GenerateAnswerAsync(question, relevantDocs);
            
            return answer;
        }
    }
}