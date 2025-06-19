// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace RagConsoleApp.Configuration
{
    public class AzureBlobStorageConfig
    {
        public string ConnectionString { get; set; } = string.Empty;
        public string ContainerName { get; set; } = string.Empty;
    }

    public class AzureAISearchConfig
    {
        public string ServiceEndpoint { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public string IndexName { get; set; } = string.Empty;
    }

    public class AzureOpenAIConfig
    {
        public string Endpoint { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public string DeploymentName { get; set; } = string.Empty;
    }

    public class AppConfig
    {
        public AzureBlobStorageConfig AzureBlobStorage { get; set; } = new();
        public AzureAISearchConfig AzureAISearch { get; set; } = new();
        public AzureOpenAIConfig AzureOpenAI { get; set; } = new();
    }
}