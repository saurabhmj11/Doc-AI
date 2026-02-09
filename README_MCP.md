# MCP Server for Ultra Doc-Intelligence

This document explains how to set up and use the MCP (Model Context Protocol) server for Ultra Doc-Intelligence.

## What is MCP?

MCP is a standardized protocol that allows AI assistants (like Claude Desktop, Cline, etc.) to interact with external tools and resources. This MCP server exposes your document intelligence capabilities so AI assistants can:

- Upload and process logistics documents
- Ask questions about documents
- Extract structured shipment data
- Manage uploaded documents

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Make sure `.env` file has your API keys:
```bash
GEMINI_API_KEY=your_key_here
```

### 3. Test the MCP Server

```bash
python run_mcp_server.py
```

The server runs via stdio and will wait for input from an MCP client.

## Connecting to MCP Clients

### Claude Desktop

1. Open Claude Desktop configuration file:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add the server configuration:
```json
{
  "mcpServers": {
    "ultra-doc-intelligence": {
      "command": "python",
      "args": [
        "f:/product/company/ultra-doc-intelligence/backend/run_mcp_server.py"
      ],
      "cwd": "f:/product/company/ultra-doc-intelligence/backend"
    }
  }
}
```

3. Restart Claude Desktop

4. You should see the server icon appear in Claude's interface

### Cline (VS Code Extension)

1. Open Cline settings in VS Code
2. Add MCP server configuration similar to above
3. Restart VS Code

## Available Tools

### 1. upload_document

Upload and process a logistics document.

**Parameters:**
- `file_path` (string, required): Absolute path to PDF, DOCX, or TXT file

**Example:**
```
User: Upload the document at f:/docs/rate_confirmation.pdf
Claude: *uses upload_document tool*
Result: Document uploaded with ID abc123
```

### 2. ask_question

Ask a question about uploaded documents.

**Parameters:**
- `document_ids` (array of strings, required): Document IDs to query
- `question` (string, required): Your question

**Example:**
```
User: What is the pickup date in document abc123?
Claude: *uses ask_question tool*
Result: The pickup is scheduled for February 10, 2026 at 10:00 AM CST.
```

### 3. extract_structured_data

Extract structured shipment information.

**Parameters:**
- `document_id` (string, required): Document to extract from

**Returns:**
- Shipment data: shipper, consignee, dates, rate, equipment, mode, etc.

### 4. list_documents

List all uploaded documents (when document registry is implemented).

### 5. delete_document

Delete an uploaded document.

**Parameters:**
- `document_id` (string, required): Document ID to delete

## Resources

### doc://documents

Lists all uploaded documents (metadata).

### doc://document/{id}

Get metadata for a specific document.

## Example Workflow

```
User: Upload f:/docs/bol_12345.pdf and tell me the shipment details

Claude (automatically):
1. Uses upload_document tool → gets document_id="xyz789"
2. Uses extract_structured_data tool → gets shipment data
3. Responds with formatted shipment information

Result: "The shipment has ID BOL-12345, pickup from ABC Logistics 
on 02/10/2026, delivery to XYZ Distribution on 02/12/2026, 
rate $2,850.00 USD."
```

## Testing with MCP Inspector

For development and testing:

```bash
npx @modelcontextprotocol/inspector python run_mcp_server.py
```

This opens a web interface where you can:
- See all available tools and resources
- Test tool execution
- View request/response logs

## Troubleshooting

### Server won't start

1. Check Python path in configuration
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Check `.env` file exists and has valid configuration

### Tools fail to execute

1. Check backend logs in `logs/api_errors.log`
2. Verify API keys are configured
3. Test the same operation via REST API first

### Claude Desktop doesn't show the server

1. Check configuration file syntax (valid JSON)
2. Restart Claude Desktop completely
3. Check Claude Desktop logs for errors

## Architecture

```
MCP Client (Claude/Cline)
    ↓ stdio
MCP Server (mcp_server.py)
    ↓ imports
FastAPI Core (routes.py, etc.)
    ↓
RAG Pipeline, Extractor, etc.
```

The MCP server is a thin wrapper that:
1. Exposes existing functionality via MCP protocol
2. Handles stdio communication with clients
3. Converts between MCP format and internal API calls

## Security Notes

- **File Access**: Only files accessible to the Python process can be uploaded
- **No Authentication**: Currently no user separation (single-user mode)
- **Local Only**: Server runs locally, no network exposure

For production use with multiple users, you'd need to add:
- Authentication/authorization
- User-specific document isolation
- File upload restrictions
- Rate limiting

## Next Steps

1. ✅ MCP server is ready to use
2. Configure in your MCP client
3. Try uploading a document and asking questions
4. Explore automated workflows with AI assistants

For issues or questions, check the main application logs or REST API documentation.
