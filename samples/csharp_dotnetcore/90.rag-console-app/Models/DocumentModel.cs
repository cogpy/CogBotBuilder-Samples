// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;

namespace RagConsoleApp.Models
{
    public class DocumentModel
    {
        [SimpleField(IsKey = true, IsFilterable = true)]
        public string Id { get; set; } = string.Empty;

        [SearchableField(IsSortable = true)]
        public string Title { get; set; } = string.Empty;

        [SearchableField(AnalyzerName = LexicalAnalyzerName.Values.EnMicrosoft)]
        public string Content { get; set; } = string.Empty;

        [SimpleField(IsFilterable = true, IsSortable = true)]
        public string FileName { get; set; } = string.Empty;

        [SimpleField(IsFilterable = true, IsSortable = true)]
        public DateTime UploadedAt { get; set; }

        [SimpleField(IsFilterable = true)]
        public string BlobUri { get; set; } = string.Empty;
    }
}