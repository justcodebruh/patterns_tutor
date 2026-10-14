import os
import re
import argparse
import unicodedata
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# PDF Page (1-indexed) = Book Page + 25
BOOK_TO_PDF_OFFSET = 25

# Part 1 Narrative Chapters + Part 2 Pattern Catalog (Book Pages)
POEAA_CATALOG = [
    # Part 1: The Narratives
    ("Narratives", "Chapter 1: Layering", 17, 24),
    ("Narratives", "Chapter 2: Organizing Domain Logic", 25, 32),
    ("Narratives", "Chapter 3: Mapping to Relational Databases", 33, 53),
    ("Narratives", "Chapter 4: Web Presentation", 55, 61),
    ("Narratives", "Chapter 5: Concurrency", 63, 80),
    ("Narratives", "Chapter 6: Session State", 81, 86),
    ("Narratives", "Chapter 7: Distribution Strategies", 87, 93),
    ("Narratives", "Chapter 8: Putting It All Together", 95, 106),
    # Chapter 9: Domain Logic Patterns
    ("Domain Logic", "Transaction Script", 110, 115),
    ("Domain Logic", "Domain Model", 116, 124),
    ("Domain Logic", "Table Module", 125, 132),
    ("Domain Logic", "Service Layer", 133, 142),
    # Chapter 10: Data Source Architectural Patterns
    ("Data Source", "Table Data Gateway", 144, 151),
    ("Data Source", "Row Data Gateway", 152, 159),
    ("Data Source", "Active Record", 160, 164),
    ("Data Source", "Data Mapper", 165, 182),
    # Chapter 11: Object-Relational Behavioral Patterns
    ("OR Behavioral", "Unit of Work", 184, 194),
    ("OR Behavioral", "Identity Map", 195, 199),
    ("OR Behavioral", "Lazy Load", 200, 214),
    # Chapter 12: Object-Relational Structural Patterns
    ("OR Structural", "Identity Field", 216, 235),
    ("OR Structural", "Foreign Key Mapping", 236, 247),
    ("OR Structural", "Association Table Mapping", 248, 261),
    ("OR Structural", "Dependent Mapping", 262, 267),
    ("OR Structural", "Embedded Value", 268, 271),
    ("OR Structural", "Serialized LOB", 272, 277),
    ("OR Structural", "Single Table Inheritance", 278, 284),
    ("OR Structural", "Class Table Inheritance", 285, 292),
    ("OR Structural", "Concrete Table Inheritance", 293, 301),
    ("OR Structural", "Inheritance Mappers", 302, 304),
    # Chapter 13: Object-Relational Metadata Mapping Patterns
    ("OR Metadata", "Metadata Mapping", 306, 315),
    ("OR Metadata", "Query Object", 316, 321),
    ("OR Metadata", "Repository", 322, 328),
    # Chapter 14: Web Presentation Patterns
    ("Web Presentation", "Model View Controller", 330, 332),
    ("Web Presentation", "Page Controller", 333, 343),
    ("Web Presentation", "Front Controller", 344, 349),
    ("Web Presentation", "Template View", 350, 360),
    ("Web Presentation", "Transform View", 361, 364),
    ("Web Presentation", "Two Step View", 365, 378),
    ("Web Presentation", "Application Controller", 379, 386),
    # Chapter 15: Distribution Patterns
    ("Distribution", "Remote Facade", 388, 400),
    ("Distribution", "Data Transfer Object", 401, 414),
    # Chapter 16: Offline Concurrency Patterns
    ("Offline Concurrency", "Optimistic Offline Lock", 416, 425),
    ("Offline Concurrency", "Pessimistic Offline Lock", 426, 437),
    ("Offline Concurrency", "Coarse-Grained Lock", 438, 448),
    ("Offline Concurrency", "Implicit Lock", 449, 454),
    # Chapter 17: Session State Patterns
    ("Session State", "Client Session State", 456, 457),
    ("Session State", "Server Session State", 458, 461),
    ("Session State", "Database Session State", 462, 464),
    # Chapter 18: Base Patterns
    ("Base", "Gateway", 466, 472),
    ("Base", "Mapper", 473, 474),
    ("Base", "Layer Supertype", 475, 475),
    ("Base", "Separated Interface", 476, 479),
    ("Base", "Registry", 480, 485),
    ("Base", "Value Object", 486, 487),
    ("Base", "Money", 488, 495),
    ("Base", "Special Case", 496, 498),
    ("Base", "Plugin", 499, 503),
    ("Base", "Service Stub", 504, 507),
    ("Base", "Record Set", 508, 510),
]

PAGE_TO_PATTERN = {
    start_pg: name
    for layer, name, start_pg, _ in POEAA_CATALOG
    if layer != "Narratives"
}


def clean_poeaa_page_text(raw_text: str) -> str:
    """Strips PDF watermarks, print codes, running headers, and soft-hyphen bugs."""
    text = unicodedata.normalize("NFKC", raw_text)  # expands ligatures like "ﬁ" -> "fi"
    text = re.sub(r"([a-zA-Z])\ufffd\s*([a-zA-Z])", r"\1\2", text)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("From the Library of"):
            continue
        if stripped.startswith("ptg10179753"):
            continue
        if re.match(r"^\d+\s+[A-Z\s]{4,}$", stripped) or re.match(r"^[A-Z\s]{4,}\s+\d+$", stripped):
            continue
        if re.match(r"^Chapter \d+$", stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def split_into_fowler_sections(pattern_text: str) -> list[tuple[str, str]]:
    """Splits a pattern's full text into (section_type, content) blocks."""
    section_regex = re.compile(
        r"^(How It Works|When to Use It|Further Reading|Example:[^\n]+)$",
        re.MULTILINE,
    )
    matches = list(section_regex.finditer(pattern_text))
    if not matches:
        return [("overview", pattern_text)]

    sections = []
    overview_text = pattern_text[: matches[0].start()].strip()
    if overview_text:
        sections.append(("overview", overview_text))

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(pattern_text)
        body = pattern_text[start_idx:end_idx].strip()

        if heading.startswith("How It Works"):
            sec_type = "how_it_works"
        elif heading.startswith("When to Use It"):
            sec_type = "when_to_use"
        elif heading.startswith("Example:"):
            sec_type = "code_example"
        else:
            sec_type = "further_reading"

        sections.append((sec_type, body))

    return sections


def extract_cross_referenced_patterns(text: str, self_pattern: str) -> str:
    """Extracts Fowler's '(116)' page citations and resolves them to pattern names."""
    cited_pages = [int(m) for m in re.findall(r"\((\d{3})\)", text)]
    related = {
        PAGE_TO_PATTERN[p]
        for p in cited_pages
        if p in PAGE_TO_PATTERN and PAGE_TO_PATTERN[p] != self_pattern
    }
    return ", ".join(sorted(related))


def get_or_build_vectorstore(
    pdf_path: str = "Patterns_Of_Enterprise_Application_Architecture.pdf",
    persist_dir: str = "./poeaa_chroma",
    force_rebuild: bool = False,
) -> Chroma:
    """Loads existing ChromaDB if already indexed, otherwise parses PDF and builds it."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(persist_dir):
        vectorstore = Chroma(
            collection_name="poeaa_patterns",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        if force_rebuild:
            # from_documents appends to an existing collection, so drop it first
            vectorstore.delete_collection()
        else:
            count = len(vectorstore.get(include=[])["ids"])
            if count > 0:
                print(f"Loaded existing ChromaDB from '{persist_dir}' ({count} chunks).")
                return vectorstore

    print(f"Parsing '{pdf_path}' and building ChromaDB at '{persist_dir}'...")
    doc = pymupdf.open(pdf_path)
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\nclass ", "\npublic ", "\n", " "],
    )

    all_documents: list[Document] = []

    for layer, pattern_name, start_book_pg, end_book_pg in POEAA_CATALOG:
        start_pdf_idx = start_book_pg + BOOK_TO_PDF_OFFSET - 1
        end_pdf_idx = end_book_pg + BOOK_TO_PDF_OFFSET - 1

        raw_pages = [
            clean_poeaa_page_text(doc[idx].get_text("text"))
            for idx in range(start_pdf_idx, min(end_pdf_idx + 1, len(doc)))
        ]
        full_pattern_text = "\n".join(raw_pages)

        for section_type, section_body in split_into_fowler_sections(full_pattern_text):
            if section_type == "further_reading":
                continue

            for chunk_text in chunk_splitter.split_text(section_body):
                header = f"[Layer: {layer} | Pattern: {pattern_name} | Section: {section_type}]\n"
                related = extract_cross_referenced_patterns(chunk_text, pattern_name)

                all_documents.append(
                    Document(
                        page_content=header + chunk_text,
                        metadata={
                            "layer": layer,
                            "pattern_name": pattern_name,
                            "section": section_type,
                            "book_pages": f"{start_book_pg}-{end_book_pg}",
                            "related_patterns": related,
                        },
                    )
                )

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        collection_name="poeaa_patterns",
        persist_directory=persist_dir,
    )
    print(f"Indexed {len(all_documents)} clean PoEAA chunks into '{persist_dir}'.")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest and query PoEAA Vector Store.")
    parser.add_argument(
        "--pdf",
        default="Patterns_Of_Enterprise_Application_Architecture.pdf",
        help="Path to the PoEAA PDF file",
    )
    parser.add_argument(
        "--persist-dir",
        default="./poeaa_chroma",
        help="Directory to persist ChromaDB",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-parsing and re-indexing the PDF",
    )
    parser.add_argument(
        "--test-pattern",
        default="Domain Model",
        help="Pattern name to run a verification query against",
    )
    args = parser.parse_args()

    # 1. Build or load the vector store
    vs = get_or_build_vectorstore(
        pdf_path=args.pdf,
        persist_dir=args.persist_dir,
        force_rebuild=args.rebuild,
    )

    # 2. Run a verification query filtering by pattern_name and section
    print(f"\n--- Verification Query for '{args.test_pattern}' (when_to_use) ---")
    results = vs.similarity_search(
        query=f"When should I use {args.test_pattern} and what are the trade-offs?",
        k=2,
        filter={
            "$and": [
                {"pattern_name": {"$eq": args.test_pattern}},
                {"section": {"$eq": "when_to_use"}},
            ]
        },
    )

    for i, doc in enumerate(results, 1):
        print(f"\n[Chunk {i}] Metadata: {doc.metadata}")
        print("-" * 60)
        print(doc.page_content[:500] + "...\n")


if __name__ == "__main__":
    main()
