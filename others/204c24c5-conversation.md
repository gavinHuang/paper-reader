# Conversation: Understanding Recursive Language Models (RLMs)

**Paper:** Recursive Language Models  
**Authors:** Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL)  
**Date:** January 4, 2026

---

## Q1: What's this paper about?

This paper introduces **Recursive Language Models (RLMs)**, a new inference strategy that allows LLMs to handle extremely long prompts (up to 10M+ tokens) - far beyond their normal context window limits.

### Key Concepts

**Main Innovation**: Instead of feeding long prompts directly into the LLM, RLMs treat the prompt as part of an external environment (specifically, a Python REPL). The model can:
- Programmatically examine and filter the prompt using code
- Recursively call itself on smaller snippets
- Process inputs 100x larger than the model's context window

### Main Findings

**Performance**: RLMs dramatically outperform base models on long-context tasks:
- Handle 2-11M token inputs effectively
- Up to 2× better performance than baselines
- Strong results even on tasks that fit within context windows (when tasks are information-dense)

**Cost**: Comparable or cheaper than alternatives, despite the recursive calls - the model selectively views context rather than ingesting everything.

### How It Works

The RLM loads the long prompt as a variable in Python, then writes code to:
1. Peek into and filter the context (e.g., using regex searches)
2. Chunk relevant sections
3. Make recursive sub-LM calls on those chunks
4. Aggregate results programmatically

The paper tests this on diverse tasks (document Q&A, code understanding, information aggregation) using GPT-5 and Qwen3-Coder, showing it successfully scales where traditional approaches fail catastrophically.

---

## Q2: So it's a predefined script that loads prompts?

Not quite - the key insight is more elegant than that:

### How RLMs Actually Work

1. **The entire long prompt is loaded as a STRING VARIABLE** in a Python REPL environment - it's NOT executed against the LLM at all initially. The prompt becomes data, like `context = "... 10M tokens of text..."`

2. **The LLM writes CODE** to interact with this variable:
   - Uses regex, string operations to search/filter the prompt
   - Programmatically extracts relevant chunks
   - Decides what sections matter based on the task

3. **Recursive sub-LLM calls on selected chunks**: When the model finds relevant sections, it can call `llm_query(chunk)` to have a sub-LLM analyze just that piece

### Key Difference

The prompt **never goes into the LLM's context window directly**. Instead:
- The LLM sees metadata (length, structure info)
- The LLM writes code to peek at small pieces
- Only selected snippets get processed by sub-LLMs

Think of it like: the LLM is an operating system that can programmatically access a large file on disk, rather than trying to load the entire file into RAM.

The paper shows examples where the model writes regex queries like `find_snippets("festival", window=200)` to search the prompt without reading it all, then recursively calls itself on just the matching sections.

---

## Q3: How does it know what the prompt is trying to do?

The RLM receives **two separate pieces**:

### 1. **The Task/Question** → Goes to the LLM normally
The user's actual query/instruction is provided to the root LLM in its prompt as usual. For example:
- "Find all documents mentioning festivals in La Union"
- "Classify each question by category"
- "What are the first and last names of the beauty pageant winner?"

### 2. **The Long Context/Data** → Loaded as a variable
Only the massive data (documents, datasets, etc.) is offloaded to the REPL environment as a string variable called `context`.

### What the Root LLM Actually Sees

From the paper's system prompt and examples, the root LLM receives something like:

```
You have access to a Python REPL environment.
The variable `context` contains [description of data].
Context length: 8,300,000 characters
Context type: List of 1000 documents

User query: [the actual question]

Write code to solve this task.
```

The LLM knows what to do because:
- It reads the user's question normally
- It sees metadata about what's in `context` 
- It writes code to explore and filter `context` based on the question
- It uses recursive sub-LM calls on relevant pieces

So the "what to do" comes from the normal prompt mechanism - only the "data to process" is offloaded to the environment.

---

## Q4: Show me an example implementation

Based on the paper's examples, here's how RLMs work in practice:

### Example: Finding Information in 1000 Documents (8.3M tokens)

#### **Input Setup**

**User Query:**
```
What are the first and last names of the person who won the beauty pageant 
at the 13th anniversary of the festival named after a vegetable stew?
```

**Context (loaded as variable):**
```python
context = [
    "Document 1: ...text...",
    "Document 2: ...text...", 
    # ... 998 more documents ...
    "Document 1000: ...text..."
]
# Total: 8,300,000 characters
```

#### **What the Root LLM Sees:**

```
You have Python REPL access with:
- Variable `context`: list of 1000 documents (8.3M chars)
- Function `llm_query(text)`: call sub-LM on small text

Task: [user query above]

Write code to solve this.
```

#### **Step-by-Step Execution**

**Iteration 1: Probe the context**

Root LLM writes code:
```python
# Search for keywords using regex
def find_snippets(keyword, window=200, max_hits=10):
    hits = []
    for i, chunk in enumerate(context):
        start = 0
        while True:
            idx = chunk.lower().find(keyword.lower(), start)
            if idx == -1: break
            s = max(0, idx - window)
            e = min(len(chunk), idx + len(keyword) + window)
            hits.append((i, chunk[s:e]))
            if len(hits) >= max_hits: return hits
            start = idx + 1
    return hits

keywords = ["festival", "beauty pageant", "13th anniversary"]
for kw in keywords:
    results = find_snippets(kw)
    print(f"Found {len(results)} matches for '{kw}'")
```

**Output:**
```
Found 3 matches for 'festival'
Found 1 matches for 'beauty pageant'
Found 1 matches for '13th anniversary'
```

**Iteration 2: Examine promising document**

Root LLM finds document at index 6 looks relevant:
```python
# Found match in document 6, query sub-LM to extract info
chunk6 = context[6]

answer = llm_query(f"""
Extract from this article:
- What festival and town?
- What year?
- Was it 13th anniversary?
- Beauty pageant winner's full name?

Article:
{chunk6}
""")

print(answer)
```

**Sub-LM response:**
```
- Festival: Dinengdeng Festival in Agoo, La Union
- Year: 2017
- 13th anniversary: Yes ("For 13 years now...")
- Winner: Maria Camille Dalmacio
```

**Iteration 3: Verify and format answer**

```python
# Verify the specific answer
final_answer = llm_query(f"""
From: {answer}
Extract ONLY first and last names of beauty pageant winner
""")

print(final_answer)  # "Maria Dalmacio"

FINAL(final_answer)  # Return as final answer
```

#### **Key Points**

1. **Context never enters LLM**: The 8.3M chars stay in Python environment
2. **Selective reading**: Only document #6 (~2K tokens) was actually processed by an LLM
3. **Cost**: $0.079 instead of $1.50-2.75 to ingest full context
4. **Recursive calls**: Sub-LM processes small chunks (2K tokens each) instead of full context

The root LLM writes code that explores and filters, then strategically calls sub-LMs on just the relevant pieces - like having a research assistant who knows how to search through files programmatically rather than reading everything.

---

## Key Takeaways

1. **Core Innovation**: Treating prompts as external data that LLMs interact with programmatically, rather than ingesting them directly
2. **Scaling**: Successfully processes 10M+ tokens (100x beyond context windows)
3. **Efficiency**: Cheaper costs through selective context viewing
4. **Performance**: 2× better than baselines on information-dense tasks
5. **Implementation**: Python REPL + recursive sub-LM calls + programmatic filtering

This approach fundamentally changes how we think about LLM context limitations - it's not about making bigger context windows, but about giving LLMs better tools to manage external information.
