// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using RagConsoleApp.Configuration;
using RagConsoleApp.Services;

namespace RagConsoleApp
{
    class Program
    {
        private static AppConfig _config;
        private static IRagService _ragService;

        static async Task Main(string[] args)
        {
            Console.WriteLine("=== RAG (Retrieval Augmented Generation) Console Application ===");
            Console.WriteLine();

            try
            {
                // Load configuration
                LoadConfiguration();

                // Validate configuration
                if (!ValidateConfiguration())
                {
                    Console.WriteLine("Please update the appsettings.json file with your Azure service configurations.");
                    return;
                }

                // Initialize services
                InitializeServices();

                // Initialize the RAG system
                await _ragService.InitializeAsync();

                // Upload sample document if it exists
                await UploadSampleDocumentIfExists();

                // Start interactive session
                await StartInteractiveSession();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
                Console.WriteLine("Please check your configuration and try again.");
            }
        }

        private static void LoadConfiguration()
        {
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build();

            _config = configuration.Get<AppConfig>();
        }

        private static bool ValidateConfiguration()
        {
            if (_config == null)
            {
                Console.WriteLine("Error: Could not load configuration.");
                return false;
            }

            var errors = new List<string>();

            if (string.IsNullOrWhiteSpace(_config.AzureBlobStorage.ConnectionString))
                errors.Add("Azure Blob Storage connection string is missing.");

            if (string.IsNullOrWhiteSpace(_config.AzureAISearch.ServiceEndpoint))
                errors.Add("Azure AI Search service endpoint is missing.");

            if (string.IsNullOrWhiteSpace(_config.AzureAISearch.ApiKey))
                errors.Add("Azure AI Search API key is missing.");

            if (string.IsNullOrWhiteSpace(_config.AzureOpenAI.Endpoint))
                errors.Add("Azure OpenAI endpoint is missing.");

            if (string.IsNullOrWhiteSpace(_config.AzureOpenAI.ApiKey))
                errors.Add("Azure OpenAI API key is missing.");

            if (string.IsNullOrWhiteSpace(_config.AzureOpenAI.DeploymentName))
                errors.Add("Azure OpenAI deployment name is missing.");

            if (errors.Any())
            {
                Console.WriteLine("Configuration errors:");
                foreach (var error in errors)
                {
                    Console.WriteLine($"- {error}");
                }
                Console.WriteLine();
                return false;
            }

            return true;
        }

        private static void InitializeServices()
        {
            var blobStorageService = new BlobStorageService(_config.AzureBlobStorage);
            var searchService = new AISearchService(_config.AzureAISearch);
            var openAIService = new OpenAIService(_config.AzureOpenAI);

            _ragService = new RagService(blobStorageService, searchService, openAIService);
        }

        private static async Task UploadSampleDocumentIfExists()
        {
            var sampleFilePath = "sample-document.txt";
            if (File.Exists(sampleFilePath))
            {
                Console.WriteLine("Sample document found. Uploading and indexing...");
                try
                {
                    var result = await _ragService.UploadAndIndexDocumentAsync(sampleFilePath);
                    Console.WriteLine(result);
                    Console.WriteLine();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error uploading sample document: {ex.Message}");
                    Console.WriteLine();
                }
            }
        }

        private static async Task StartInteractiveSession()
        {
            Console.WriteLine("RAG system is ready! You can now ask questions.");
            Console.WriteLine("Available commands:");
            Console.WriteLine("- Ask any question to get an AI-generated answer based on indexed documents");
            Console.WriteLine("- Type 'upload <filepath>' to upload and index a new document");
            Console.WriteLine("- Type 'exit' to quit");
            Console.WriteLine();

            while (true)
            {
                Console.Write("You: ");
                var input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input))
                    continue;

                if (input.ToLower() == "exit")
                {
                    Console.WriteLine("Goodbye!");
                    break;
                }

                if (input.ToLower().StartsWith("upload "))
                {
                    var filePath = input.Substring(7).Trim();
                    await HandleDocumentUpload(filePath);
                }
                else
                {
                    await HandleQuestion(input);
                }

                Console.WriteLine();
            }
        }

        private static async Task HandleDocumentUpload(string filePath)
        {
            try
            {
                var result = await _ragService.UploadAndIndexDocumentAsync(filePath);
                Console.WriteLine($"Assistant: {result}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Assistant: Error uploading document: {ex.Message}");
            }
        }

        private static async Task HandleQuestion(string question)
        {
            try
            {
                var answer = await _ragService.AnswerQuestionAsync(question);
                Console.WriteLine($"Assistant: {answer}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Assistant: Error generating answer: {ex.Message}");
            }
        }
    }
}