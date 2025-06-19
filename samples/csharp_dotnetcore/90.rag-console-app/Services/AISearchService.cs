// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using RagConsoleApp.Configuration;
using RagConsoleApp.Models;

namespace RagConsoleApp.Services
{
    public interface IAISearchService
    {
        Task InitializeIndexAsync();
        Task IndexDocumentAsync(DocumentModel document);
        Task<List<DocumentModel>> SearchDocumentsAsync(string query, int top = 3);
    }

    public class AISearchService : IAISearchService
    {
        private readonly SearchIndexClient _indexClient;
        private readonly SearchClient _searchClient;
        private readonly AzureAISearchConfig _config;

        public AISearchService(AzureAISearchConfig config)
        {
            _config = config;
            var credential = new AzureKeyCredential(_config.ApiKey);
            _indexClient = new SearchIndexClient(new Uri(_config.ServiceEndpoint), credential);
            _searchClient = new SearchClient(new Uri(_config.ServiceEndpoint), _config.IndexName, credential);
        }

        public async Task InitializeIndexAsync()
        {
            try
            {
                // Check if index exists
                await _indexClient.GetIndexAsync(_config.IndexName);
                Console.WriteLine($"Search index '{_config.IndexName}' already exists.");
            }
            catch (RequestFailedException ex) when (ex.Status == 404)
            {
                // Create the index
                var searchIndex = new SearchIndex(_config.IndexName)
                {
                    Fields = new FieldBuilder().Build(typeof(DocumentModel))
                };

                await _indexClient.CreateIndexAsync(searchIndex);
                Console.WriteLine($"Search index '{_config.IndexName}' created successfully.");
            }
        }

        public async Task IndexDocumentAsync(DocumentModel document)
        {
            var batch = IndexDocumentsBatch.Create(IndexDocumentsAction.Upload(document));
            await _searchClient.IndexDocumentsAsync(batch);
            
            // Wait a moment for indexing to complete
            await Task.Delay(1000);
        }

        public async Task<List<DocumentModel>> SearchDocumentsAsync(string query, int top = 3)
        {
            var searchOptions = new SearchOptions
            {
                Size = top,
                IncludeTotalCount = true,
                SearchFields = { "Title", "Content" },
                Select = { "Id", "Title", "Content", "FileName", "UploadedAt", "BlobUri" }
            };

            var response = await _searchClient.SearchAsync<DocumentModel>(query, searchOptions);
            var results = new List<DocumentModel>();

            await foreach (var result in response.Value.GetResultsAsync())
            {
                results.Add(result.Document);
            }

            return results;
        }
    }
}