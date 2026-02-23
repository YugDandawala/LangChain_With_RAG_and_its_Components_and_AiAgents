<h1 align="center">🚀 LangChain RAG – Complete Learning Repository</h1>

<p align="center">
A structured, beginner-friendly, and deeply informative guide to learning 
<strong>LangChain, RAG (Retrieval-Augmented Generation), Embeddings, Chains, Runnables, 
Structured Output, and AI Model Workflows</strong>.
</p>

<hr>

<h2>📌 Table of Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#models">Models</a></li>
  <li><a href="#prompts">Prompts</a></li>
  <li><a href="#output-source">Output Source & Structured Output</a></li>
  <li><a href="#output-parsers">Output Parsers</a></li>
  <li><a href="#chains">Chains</a></li>
  <li><a href="#runnables">Runnables</a></li>
  <li><a href="#rag-pipeline">RAG Pipeline</a></li>
  <li><a href="#rag-components">RAG Components</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#troubleshooting">Troubleshooting</a></li>
  <li><a href="#contributors">Contributors</a></li>
  <li><a href="#license">License</a></li>
</ul>

<hr>

<h2 id="introduction">📘 Introduction</h2>
<p>This repository is a complete guide for learning and practicing:</p>

<ul>
  <li>🔹 LangChain Core Concepts</li>
  <li>🔹 RAG (Retrieval-Augmented Generation)</li>
  <li>🔹 Text Loading & Processing</li>
  <li>🔹 Vector Stores & Semantic Search</li>
  <li>🔹 Embeddings</li>
  <li>🔹 Chains & Runnables</li>
  <li>🔹 Structured Output Parsing</li>
</ul>

<p>
You will find in-depth explanations, practical examples, and hands-on scripts 
for building real-world LangChain & RAG systems from scratch.
</p>

<hr>

<h2 id="models">🧠 Models</h2>

<h3>1️⃣ Language Models (LLMs)</h3>
<ul>
  <li>Text generation</li>
  <li>Summarization</li>
  <li>Translation</li>
  <li>Coding</li>
  <li>Creative writing</li>
</ul>

<h3>💬 Chat Models</h3>
<ul>
  <li>Optimized for dialogue</li>
  <li>Maintain context</li>
  <li>Understand conversational roles</li>
</ul>

<h3>📍 HuggingFace</h3>
<p>A major hub for open-source AI models.</p>

<h3>2️⃣ Embedding Models</h3>
<p>Convert text → vectors for semantic search.</p>

<hr>

<h2 id="prompts">✍️ Prompts</h2>

<ul>
  <li><strong>Text-based Prompts</strong>
    <ul>
      <li>Static prompts</li>
      <li>Dynamic prompts</li>
      <li>Single or multiple messages</li>
    </ul>
  </li>
  <li><strong>Multimodal Prompts</strong> (images, audio, etc.)</li>
</ul>

<hr>

<h2 id="output-source">📤 Output Source & Structured Output — Instruction Guide</h2>

<h3>1️⃣ LLM Output Types</h3>
<p>
LLMs naturally generate <strong>unstructured text</strong>. To use results in applications, we need structured output such as JSON or Pydantic.
</p>

<ul>
  <li><strong>Default / Unstructured Output</strong></li>
  <li><strong>Structured Output</strong></li>
</ul>

<h3>Structured Output Needed For:</h3>
<ul>
  <li>Data extraction</li>
  <li>API development</li>
  <li>Agents</li>
  <li>Databases</li>
  <li>Automation pipelines</li>
</ul>

<hr>

<h3>2️⃣ Why Structured Output Is Needed</h3>

<p>Systems require predictable machine-readable formats like JSON.</p>

<ul>
  <li>Avoid invalid formatting</li>
  <li>Enforce consistency</li>
  <li>Prevent pipeline failures</li>
</ul>

<hr>

<h3>3️⃣ Supported Structured Output Formats</h3>

<ul>
  <li><strong>TypedDict</strong></li>
  <li><strong>Pydantic Models</strong></li>
  <li><strong>JSON Schema</strong></li>
</ul>

<hr>

<h2 id="output-parsers">🔧 Output Parsers</h2>

<ul>
  <li><strong>String Output Parser</strong></li>
  <li><strong>JSON Output Parser</strong></li>
  <li><strong>Pydantic Output Parser</strong></li>
  <li><strong>Structured Output Parser (Legacy)</strong></li>
</ul>

<p>Parsers ensure consistency, type safety, and reliability.</p>

<hr>

<h3>5️⃣ Workflow</h3>

<pre>
User Input
 ↓
Prompt (with format instructions)
 ↓
LLM (raw output)
 ↓
Parser (JSON/Pydantic)
 ↓
Structured Output
</pre>

<hr>

<h2 id="chains">⛓️ Chains</h2>

<p>Chains allow multi-step pipelines in LangChain.</p>

<ul>
  <li><strong>Sequential Chains</strong></li>
  <li><strong>Parallel Chains</strong></li>
  <li><strong>Conditional Chains</strong></li>
</ul>

<pre><code>Prompt → Model → Parser</code></pre>

<hr>

<h2 id="runnables">⚙️ Runnables</h2>

<p>Modern building blocks for LangChain pipelines.</p>

<h4>🟦 Primitive Runnables</h4>
<ul>
  <li>RunnableSequence</li>
  <li>RunnableParallel</li>
  <li>RunnablePassthrough</li>
  <li>RunnableLambda</li>
  <li>RunnableBranch</li>
</ul>

<h4>🟩 Task-Specific Runnables</h4>
<ul>
  <li>PromptTemplate</li>
  <li>LLM / Chat Model</li>
  <li>Retriever</li>
  <li>Parser</li>
</ul>

<hr>

<h2 id="rag-pipeline">🟣 RAG Pipeline</h2>

<p>
RAG = <strong>Retrieval-Augmented Generation</strong> — the model retrieves context before generating output.
</p>

<hr>

<h2 id="rag-components">🧩 RAG Components</h2>

<h3>1️⃣ Document Loaders</h3>
<ul>
  <li>TXT</li>
  <li>PDF</li>
  <li>Directory</li>
  <li>CSV</li>
  <li>Web pages</li>
</ul>

<h3>2️⃣ Text Splitters</h3>
<ul>
  <li>Length-based</li>
  <li>Structure-based</li>
  <li>Semantic-based</li>
</ul>

<h3>3️⃣ Vector Stores</h3>

<p>Store embeddings for similarity search.</p>

<h4>Use cases:</h4>
<ul>
  <li>Semantic search</li>
  <li>RAG</li>
  <li>Recommendations</li>
</ul>

<h3>4️⃣ Retrievers</h3>
<ol>
  <li>Wikipedia Retriever</li>
  <li>Vector Store Retriever</li>
  <li>MMR Retriever</li>
  <li>Multi-query Retriever</li>
  <li>Contextual Compression Retriever</li>
</ol>

<hr>

<h2 id="installation">📦 Installation</h2>

<pre><code>git clone &lt;your-repo-url&gt;
cd your-repo

pip install -r requirements.txt
</code></pre>

<hr>

<h2 id="troubleshooting">🛠 Troubleshooting</h2>

<ul>
  <li>Use Python 3.10+</li>
  <li>Set your API keys correctly</li>
  <li>Ensure LangChain version compatibility</li>
</ul>

<hr>

<h2 id="contributors">👨‍💻 Contributors</h2>
<p>This repository is maintained and documented by <strong>Yug Dandawala</strong>.</p>

<hr>

<h2 id="license">📄 License</h2>
<p>Free to use, modify, and share.</p>
