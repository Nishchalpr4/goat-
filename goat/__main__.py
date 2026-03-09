"""GOAT CLI — command-line interface for the Growth-Oriented Analysis Toolkit.

Usage:
    python -m goat query "What drives AAPL margin expansion?"
    python -m goat ingest --file report.txt --type 10-K --ticker AAPL
    python -m goat ingest --companies companies.json
    python -m goat company AAPL --report
    python -m goat compare AAPL MSFT GOOGL
    python -m goat status
"""

import argparse
import sys
import json
import logging
from pathlib import Path

from goat.config import Config, default_config


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_query(args, config: Config) -> None:
    """Execute a query."""
    from goat.query.pipeline import QueryPipeline
    from goat.reporting.terminal import TerminalReporter

    pipeline = QueryPipeline(config)
    reporter = TerminalReporter()

    query_text = " ".join(args.query)
    response = pipeline.process(query_text)
    reporter.print_query_response(response)

    if args.json:
        from goat.reporting.json_export import JSONExporter
        exporter = JSONExporter()
        data = exporter.export_query_response(response)
        print(json.dumps(data, indent=2))


def cmd_ingest(args, config: Config) -> None:
    """Ingest documents or company data."""
    from goat.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(config)

    if args.companies:
        stats = pipeline.ingest_company_batch(args.companies)
        print(f"Ingested {stats.documents_processed} companies "
              f"({stats.documents_failed} failed)")
    elif args.file:
        metadata = {
            "doc_type": args.type or "news",
            "ticker": args.ticker or "",
            "period": args.period or "",
        }
        stats = pipeline.ingest_file(args.file, metadata)
        print(f"Ingested: {stats.chunks_created} chunks, "
              f"{stats.embeddings_generated} embeddings, "
              f"{stats.total_time_ms:.0f}ms")
    else:
        print("Specify --file or --companies")
        sys.exit(1)

    if stats.errors:
        for err in stats.errors:
            print(f"  ERROR: {err}")
    if stats.warnings:
        for warn in stats.warnings:
            print(f"  WARN: {warn}")


def cmd_company(args, config: Config) -> None:
    """Display company information."""
    from goat.models.company import Company, Identifier, CompanyMetadata
    from goat.reporting.terminal import TerminalReporter

    reporter = TerminalReporter()

    # In a full implementation, this would look up the company from storage.
    # For now, create a placeholder.
    ticker = args.ticker
    company = Company(
        canonical_id=ticker,
        name=ticker,
        identifiers=[Identifier(id_type="ticker", value=ticker)],
        metadata=CompanyMetadata(),
    )
    reporter.print_company_overview(company)


def cmd_compare(args, config: Config) -> None:
    """Compare multiple companies."""
    from goat.models.company import Company, Identifier, CompanyMetadata
    from goat.reporting.terminal import TerminalReporter

    reporter = TerminalReporter()
    companies = []
    for ticker in args.tickers:
        companies.append(Company(
            canonical_id=ticker,
            name=ticker,
            identifiers=[Identifier(id_type="ticker", value=ticker)],
            metadata=CompanyMetadata(),
        ))
    reporter.print_comparison_table(companies, {})


def cmd_status(args, config: Config) -> None:
    """Show system status."""
    from goat.operations.monitoring import MonitoringService, LatencyStats
    from goat.reporting.terminal import TerminalReporter

    monitor = MonitoringService()
    reporter = TerminalReporter()
    reporter.print_system_status(
        monitor.get_latency_stats(),
        monitor.get_quality_snapshot(),
    )


def cmd_eval(args, config: Config) -> None:
    """Run offline evaluation."""
    from goat.operations.evaluation import EvaluationHarness

    harness = EvaluationHarness()

    if args.dataset:
        queries = harness.load_beir_dataset(
            args.dataset + "/queries.tsv",
            args.dataset + "/qrels.tsv",
        )
        print(f"Loaded {len(queries)} evaluation queries")
    else:
        print("Specify --dataset path")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="goat",
        description="GOAT — Growth-Oriented Analysis Toolkit",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # query
    p_query = subparsers.add_parser("query", help="Execute an analysis query")
    p_query.add_argument("query", nargs="+", help="Query text")
    p_query.add_argument("--json", action="store_true",
                          help="Output as JSON")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("--file", type=str, help="File to ingest")
    p_ingest.add_argument("--companies", type=str,
                           help="JSON file with company data")
    p_ingest.add_argument("--type", type=str, default="news",
                           help="Document type (10-K, 10-Q, transcript, news)")
    p_ingest.add_argument("--ticker", type=str, help="Company ticker")
    p_ingest.add_argument("--period", type=str, help="Filing period")

    # company
    p_company = subparsers.add_parser("company", help="Show company info")
    p_company.add_argument("ticker", help="Company ticker")
    p_company.add_argument("--report", action="store_true",
                            help="Full report")

    # compare
    p_compare = subparsers.add_parser("compare",
                                       help="Compare companies")
    p_compare.add_argument("tickers", nargs="+", help="Tickers to compare")

    # status
    subparsers.add_parser("status", help="System status")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation")
    p_eval.add_argument("--dataset", type=str, help="Dataset directory")

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = default_config

    commands = {
        "query": cmd_query,
        "ingest": cmd_ingest,
        "company": cmd_company,
        "compare": cmd_compare,
        "status": cmd_status,
        "eval": cmd_eval,
    }

    if args.command in commands:
        commands[args.command](args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
