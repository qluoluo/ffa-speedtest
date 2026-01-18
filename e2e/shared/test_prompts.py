#!/usr/bin/env python3
"""Test prompts for end-to-end benchmarking."""

# Short prompt for quick tests
SHORT_PROMPT = """The quick brown fox jumps over the lazy dog. This is a simple test prompt to verify that the model is working correctly."""

# Medium prompt (~500 tokens)
MEDIUM_PROMPT = """
In the realm of artificial intelligence, large language models have emerged as a transformative technology
that is reshaping how we interact with machines and process information. These models, trained on vast
corpora of text data, have demonstrated remarkable capabilities in understanding and generating human-like
text across a wide range of domains and tasks.

The architecture of modern large language models typically builds upon the transformer architecture,
first introduced in the seminal "Attention Is All You Need" paper. This architecture relies on
self-attention mechanisms that allow the model to weigh the importance of different parts of the input
when generating each output token. The key innovation of transformers is their ability to process
sequences in parallel, rather than sequentially like previous recurrent neural network architectures.

Training these models requires enormous computational resources. The largest models today contain
hundreds of billions of parameters and are trained on datasets containing trillions of tokens.
This training process involves optimizing the model to predict the next token in a sequence,
a task known as language modeling. Through this seemingly simple objective, the models learn
complex patterns of language, world knowledge, and even reasoning capabilities.

One of the most exciting developments in this field is the emergence of instruction-following
capabilities. By fine-tuning pre-trained language models on datasets of human-written instructions
and responses, researchers have created models that can follow complex multi-step instructions,
engage in dialogue, and assist users with a wide variety of tasks.

However, these models also face significant challenges. They can generate plausible-sounding but
incorrect information, a phenomenon known as "hallucination." They may also exhibit biases present
in their training data and can be sensitive to the exact phrasing of prompts. Researchers are
actively working on techniques to mitigate these issues and make language models more reliable,
truthful, and aligned with human values.

The question we must now answer is:
"""

# Long prompt for stress testing (~2000 tokens)
LONG_PROMPT = """
The history of computing spans several centuries, from the earliest mechanical calculators to the
sophisticated quantum computers being developed today. This journey represents one of humanity's
greatest intellectual achievements, fundamentally transforming every aspect of modern life.

The story begins in the 17th century with pioneers like Blaise Pascal and Gottfried Wilhelm Leibniz,
who created some of the first mechanical calculators. Pascal's Pascaline, developed in 1642, could
perform addition and subtraction, while Leibniz's stepped reckoner extended these capabilities to
multiplication and division. These early devices laid the conceptual groundwork for automated computation.

The 19th century saw significant advances with Charles Babbage's designs for the Difference Engine
and the more ambitious Analytical Engine. While never fully completed in his lifetime, the Analytical
Engine incorporated many concepts that would later become fundamental to computing: a central processor,
memory, and the ability to be programmed using punched cards. Ada Lovelace's notes on the Analytical
Engine are often considered the first computer programs, making her history's first programmer.

The electromechanical era began in the late 19th century with Herman Hollerith's tabulating machines,
which used punched cards to process the 1890 US Census data. This technology evolved into IBM's
early business machines and laid the foundation for the data processing industry.

The 1930s and 1940s marked the transition to electronic computing. Alan Turing's theoretical work
on computability provided the mathematical foundations, while engineers and scientists around the
world began building electronic computers. ENIAC, completed in 1945, is often considered the first
general-purpose electronic computer. It contained over 17,000 vacuum tubes and could perform
calculations thousands of times faster than any previous machine.

The invention of the transistor at Bell Labs in 1947 revolutionized electronics and computing.
Transistors were smaller, faster, more reliable, and consumed less power than vacuum tubes.
This led to the second generation of computers in the late 1950s and early 1960s.

The integrated circuit, developed independently by Jack Kilby and Robert Noyce in 1958-1959,
enabled the miniaturization that continues to this day. Gordon Moore's 1965 observation that
the number of transistors on a chip doubles approximately every two years—now known as Moore's
Law—has guided the semiconductor industry for decades.

The personal computer revolution began in the 1970s with machines like the Altair 8800 and
the Apple I. Microsoft and Apple, founded in this era, would become two of the most influential
technology companies in history. The IBM PC, introduced in 1981, established standards that
still influence computing today.

The rise of the Internet transformed computing from an isolated activity to a globally connected
experience. ARPANET, developed in the late 1960s, evolved into the Internet we know today.
Tim Berners-Lee's invention of the World Wide Web in 1989 made the Internet accessible to
billions of people and created entirely new industries and ways of life.

The 21st century has seen the emergence of mobile computing, cloud computing, and artificial
intelligence as dominant paradigms. Smartphones have put powerful computers in billions of pockets.
Cloud computing has shifted processing power from local machines to vast data centers. And
artificial intelligence, particularly machine learning and deep learning, has enabled computers
to perform tasks once thought to require human intelligence.

The field of quantum computing represents the next frontier. By harnessing quantum mechanical
phenomena like superposition and entanglement, quantum computers promise to solve certain
problems exponentially faster than classical computers. Companies and research institutions
around the world are racing to build practical quantum computers.

Looking back at this remarkable history, we can see that computing has always been driven by
the human desire to automate mental labor and extend our cognitive capabilities. Each generation
of computing technology has built upon the achievements of its predecessors while opening new
possibilities that earlier generations could not have imagined.

As we stand at the threshold of new developments in quantum computing, artificial intelligence,
and other emerging technologies, we can be certain that the history of computing is far from
complete. The next chapters will be written by researchers, engineers, and entrepreneurs who
continue to push the boundaries of what machines can do.

The fundamental question that has driven computing from its earliest days remains as relevant
as ever: How can we build machines that extend human capabilities and help us solve the great
challenges facing humanity?

Based on this comprehensive overview of computing history, please provide your analysis of:
"""


def get_test_prompts():
    """Return a dictionary of test prompts."""
    return {
        "short": SHORT_PROMPT,
        "medium": MEDIUM_PROMPT,
        "long": LONG_PROMPT,
    }


def generate_repeated_prompt(base_prompt: str, target_tokens: int, tokenizer) -> str:
    """Generate a prompt with approximately target_tokens by repeating the base prompt."""
    current_tokens = len(tokenizer.encode(base_prompt))

    if current_tokens >= target_tokens:
        return base_prompt

    # Repeat the prompt until we reach the target
    repeated = base_prompt
    while current_tokens < target_tokens:
        repeated = repeated + "\n\n" + base_prompt
        current_tokens = len(tokenizer.encode(repeated))

    return repeated
