from enum import Enum
import logging
from typing import Any
import json

# Import MCP server
from mcp.server.models import InitializationOptions
from mcp.types import (
    TextContent,
    Tool,
    Resource,
    Prompt,
    PromptArgument,
    GetPromptResult,
    PromptMessage,
)
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

logger = logging.getLogger(__name__)
logger.info("Starting deep research server")


### Prompt templates
class DeepResearchPrompts(str, Enum):
    DEEP_RESEARCH = "deep-research"


class PromptArgs(str, Enum):
    RESEARCH_QUESTION = "research_question"


PROMPT_TEMPLATE = """
You are a researcher exploring: {research_question}

Your task is iterative discovery - not template filling. Follow curiosity, not checklists.

## CRITICAL: Use Real Research Tools, Not AI Consultation

You MUST perform actual research using concrete tools:
- WebSearch: For finding information on the internet
- WebFetch: For extracting content from specific URLs
- File analysis: For examining code, documentation, data

DO NOT:
- "Consult" or "ask" other AIs for their thoughts
- Simulate research by imagining what sources might say
- Generate hypothetical findings without actual searches
- Use phrases like "I'll consult with a peer AI" or "Let me ask another agent"

Every finding must come from a real source you've actually searched for and read.

CORRECT approach:
"I'll search for information about quantum computing applications"
→ Uses WebSearch with query "quantum computing real world applications 2024"
→ Finds arxiv.org paper, uses WebFetch to read it
→ Cites specific findings: "According to arxiv.org/abs/2024.xxxxx..."

WRONG approach:
"Let me consult with an AI specialized in quantum computing"
"I'll ask another agent what they know about this"
"Based on what other AIs might say about quantum computing..."

## Research Process

**Maintain a living question queue:**
Start by creating a list of initial questions about your topic. As you research:
- Add new questions that emerge from your findings
- Mark questions as answered when sufficiently explored
- Track the depth level of each question (initial = level 1, questions spawned from level 1 = level 2, etc.)
- Continue until your queue is empty AND no new questions emerge

This queue is your compass - you're not done until it's exhausted.

**Initial exploration:**
- Map the landscape. What are the key concepts, actors, and relationships?
- Generate 5-10 initial subquestions for your queue (these are level 1)
- Identify which threads seem most promising to pursue first

**MANDATORY iterative deepening:**
You MUST go at least 3 levels deep. Most initial research stops at level 1 - this is unacceptable.

Level 1: Your initial questions
Level 2: Questions that arise from level 1 answers (MANDATORY - each level 1 question must spawn 2-3 level 2 questions)
Level 3: Questions that arise from level 2 answers (MANDATORY - at least half of level 2 questions must spawn level 3 questions)
Level 4+: Continue if the topic demands it

Process - PARALLELIZE FOR SPEED:
1. After generating initial questions, group them by independence (questions that don't depend on each other's answers)
2. Launch MULTIPLE research tasks IN PARALLEL using the Task tool:
   - Each task MUST be instructed to use WebSearch and WebFetch for actual research
   - Each parallel task takes 2-3 independent questions
   - Run 3-5 research tasks simultaneously when possible
   - Explicitly tell each task: "Use WebSearch to find information, then WebFetch to read sources"
   - This dramatically reduces research time and keeps each task's context focused
3. When tasks complete, synthesize their findings and generate level 2 questions
4. Again, group independent questions and launch parallel research tasks
5. Continue this parallel exploration pattern through all depth levels

For EACH research task - USE ACTUAL RESEARCH TOOLS:
1. Pick assigned questions from the queue
2. MANDATORY - Perform REAL research using these tools:
   - WebSearch: Search the internet for current information, academic papers, documentation
   - WebFetch: Extract detailed content from specific URLs found during search
   - File reading: Analyze code repositories, documentation, technical specs
   - DO NOT just "consult" or "ask" other AIs for their opinions
   - DO NOT simulate research or make up findings
3. Gather PRIMARY SOURCES:
   - Academic papers, technical documentation, official websites
   - Code repositories, API documentation, specifications
   - News articles, research reports, data sets
   - Expert interviews, conference talks, lectures (via transcripts)
4. MANDATORY: Generate 2-3 follow-up questions from EVERY answer:
   - "This source mentions X - what exactly is X? What are X's implications?"
   - "These experts disagree on Y - why? What evidence supports each side?"
   - "This claims Z - what's the mechanism? What are the edge cases?"
   - "This pattern emerges - does it hold in other contexts? What are the exceptions?"
5. Return concrete findings with source citations to main research process

CRITICAL: Use parallel research tasks with REAL web searches and data gathering. You must use WebSearch, WebFetch, and other concrete research tools. DO NOT simulate research by consulting AI peers - that's not research, it's speculation. If you're not actively searching and reading real sources, you're doing it wrong.

WARNING: If you find yourself with fewer than 15 total questions across all levels, you're not going deep enough. Real research is fractal - every answer opens multiple new doors.

VERIFICATION CHECKLIST - Before proceeding to the next level:
- Have I performed at least 3 WebSearch queries per question?
- Have I used WebFetch to read at least 2 sources per question?
- Can I cite specific URLs for every claim I make?
- Have I found primary sources, not just AI-generated summaries?
If any answer is NO, you haven't done real research yet.

**Source evaluation:**
- Not all sources are equal. Note credibility, recency, and potential biases
- When sources contradict, investigate why - often the most interesting insights lie here
- Primary sources > secondary sources > opinion pieces

**Synthesis focus:**
Don't just collect facts - connect them:
- How do findings relate to each other?
- What patterns emerge across sources?
- Where do experts disagree and why?
- What remains unknown or uncertain?

## Writing Your Findings

Let structure emerge from content, not vice versa. Your research might naturally organize as:
- A narrative journey through evolving understanding
- Technical deep-dive with code examples and architecture details
- Comparative analysis with side-by-side evaluation
- Historical progression showing how thinking evolved
- Problem-solution exploration
- Or something else entirely

Write concisely. Every sentence should add value. No padding, no boilerplate sections.

Good research writing:
- States findings clearly with evidence
- Acknowledges contradictions and uncertainties
- Cites sources inline naturally (not in a bibliography dump)
- Uses formatting (tables, lists, code blocks) only when it clarifies
- Stops when the essential is communicated

Bad research writing:
- Forces findings into predetermined sections
- Includes "executive summaries" that just repeat content
- Lists "key benefits" or "success metrics" unnecessarily
- Adds "next steps" when not requested
- Uses passive voice to sound academic

## What Not to Do

- Don't write sections just because they're traditional (no forced "Introduction", "Methodology", "Conclusion")
- Don't claim comprehensiveness - all research has boundaries
- Don't hide uncertainty behind confident language
- Don't make up information to fill gaps
- Don't write meta-commentary about your research process

Begin. Follow threads. Build understanding. Write what matters.
"""


### Research Processor
class ResearchProcessor:
    def __init__(self):
        self.research_data = {
            "question": "",
            "elaboration": "",
            "subquestions": [],
            "search_results": {},
            "extracted_content": {},
            "final_report": "",
        }
        self.notes: list[str] = []

    def add_note(self, note: str):
        """Add a note to the research process."""
        self.notes.append(note)
        logger.debug(f"Note added: {note}")

    def update_research_data(self, key: str, value: Any):
        """Update a specific key in the research data dictionary."""
        self.research_data[key] = value
        self.add_note(f"Updated research data: {key}")

    def get_research_notes(self) -> str:
        """Return all research notes as a newline-separated string."""
        return "\n".join(self.notes)

    def get_research_data(self) -> dict:
        """Return the current research data dictionary."""
        return self.research_data


### MCP Server Definition
async def main():
    research_processor = ResearchProcessor()
    server = Server("deep-research-server")

    @server.list_resources()
    async def handle_list_resources() -> list[Resource]:
        logger.debug("Handling list_resources request")
        return [
            Resource(
                uri="research://notes",
                name="Research Process Notes",
                description="Notes generated during the research process",
                mimeType="text/plain",
            ),
            Resource(
                uri="research://data",
                name="Research Data",
                description="Structured data collected during the research process",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> str:
        logger.debug(f"Handling read_resource request for URI: {uri}")
        if str(uri) == "research://notes":
            return research_processor.get_research_notes()
        elif str(uri) == "research://data":
            return json.dumps(research_processor.get_research_data(), indent=2)
        else:
            raise ValueError(f"Unknown resource: {uri}")

    @server.list_prompts()
    async def handle_list_prompts() -> list[Prompt]:
        logger.debug("Handling list_prompts request")
        return [
            Prompt(
                name=DeepResearchPrompts.DEEP_RESEARCH,
                description="A prompt to conduct deep research on a question",
                arguments=[
                    PromptArgument(
                        name=PromptArgs.RESEARCH_QUESTION,
                        description="The research question to investigate",
                        required=True,
                    ),
                ],
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        logger.debug(f"Handling get_prompt request for {name} with args {arguments}")
        if name != DeepResearchPrompts.DEEP_RESEARCH:
            logger.error(f"Unknown prompt: {name}")
            raise ValueError(f"Unknown prompt: {name}")

        if not arguments or PromptArgs.RESEARCH_QUESTION not in arguments:
            logger.error("Missing required argument: research_question")
            raise ValueError("Missing required argument: research_question")

        research_question = arguments[PromptArgs.RESEARCH_QUESTION]
        prompt = PROMPT_TEMPLATE.format(research_question=research_question)

        # Store the research question
        research_processor.update_research_data("question", research_question)
        research_processor.add_note(
            f"Research initiated on question: {research_question}"
        )

        logger.debug(
            f"Generated prompt template for research_question: {research_question}"
        )
        return GetPromptResult(
            description=f"Deep research template for: {research_question}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt.strip()),
                )
            ],
        )

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        logger.debug("Handling list_tools request")
        # We're not exposing any tools since we'll be using Claude's built-in web search
        return []

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.debug("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="deep-research-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
