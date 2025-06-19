// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Azure.AI.OpenAI;
using Azure;
using RagConsoleApp.Configuration;
using RagConsoleApp.Models;

namespace RagConsoleApp.Services
{
    public interface IOpenAIService
    {
        Task<string> GenerateAnswerAsync(string question, List<DocumentModel> context);
    }

    public class OpenAIService : IOpenAIService
    {
        private readonly OpenAIClient _client;
        private readonly AzureOpenAIConfig _config;

        public OpenAIService(AzureOpenAIConfig config)
        {
            _config = config;
            _client = new OpenAIClient(new Uri(_config.Endpoint), new AzureKeyCredential(_config.ApiKey));
        }

        public async Task<string> GenerateAnswerAsync(string question, List<DocumentModel> context)
        {
            var contextText = string.Join("\n\n", context.Select(doc => 
                $"Document: {doc.Title}\nContent: {doc.Content}"));

            var systemMessage = @"You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. 
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise but thorough in your response.";

            var userMessage = $@"Context:
{contextText}

Question: {question}

Please provide a clear answer based on the context above.";

            var chatCompletionsOptions = new ChatCompletionsOptions(_config.DeploymentName, new List<ChatRequestMessage>
            {
                new ChatRequestSystemMessage(systemMessage),
                new ChatRequestUserMessage(userMessage)
            })
            {
                Temperature = 0.3f,
                MaxTokens = 500
            };

            var response = await _client.GetChatCompletionsAsync(chatCompletionsOptions);
            
            return response.Value.Choices[0].Message.Content;
        }
    }
}