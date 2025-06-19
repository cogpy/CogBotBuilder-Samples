// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Threading.Tasks;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using RagConsoleApp.Configuration;

namespace RagConsoleApp.Services
{
    public interface IBlobStorageService
    {
        Task<string> UploadDocumentAsync(string fileName, string content);
        Task<string> GetDocumentContentAsync(string fileName);
        Task<bool> DocumentExistsAsync(string fileName);
    }

    public class BlobStorageService : IBlobStorageService
    {
        private readonly BlobServiceClient _blobServiceClient;
        private readonly BlobContainerClient _containerClient;
        private readonly AzureBlobStorageConfig _config;

        public BlobStorageService(AzureBlobStorageConfig config)
        {
            _config = config;
            _blobServiceClient = new BlobServiceClient(_config.ConnectionString);
            _containerClient = _blobServiceClient.GetBlobContainerClient(_config.ContainerName);
        }

        public async Task<string> UploadDocumentAsync(string fileName, string content)
        {
            // Ensure container exists
            await _containerClient.CreateIfNotExistsAsync(PublicAccessType.None);

            // Upload the document
            var blobClient = _containerClient.GetBlobClient(fileName);
            using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(content));
            
            await blobClient.UploadAsync(stream, overwrite: true);
            
            return blobClient.Uri.ToString();
        }

        public async Task<string> GetDocumentContentAsync(string fileName)
        {
            var blobClient = _containerClient.GetBlobClient(fileName);
            
            if (!await blobClient.ExistsAsync())
            {
                throw new FileNotFoundException($"Document '{fileName}' not found in blob storage.");
            }

            var response = await blobClient.DownloadContentAsync();
            return response.Value.Content.ToString();
        }

        public async Task<bool> DocumentExistsAsync(string fileName)
        {
            var blobClient = _containerClient.GetBlobClient(fileName);
            return await blobClient.ExistsAsync();
        }
    }
}