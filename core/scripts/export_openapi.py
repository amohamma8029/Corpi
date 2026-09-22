import argparse
import json

from core.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Corpi core OpenAPI schema.")
    parser.add_argument("output", help="Path of the JSON file to write.")
    args = parser.parse_args()

    app = create_app(token="schema-export-only")
    schema = app.openapi()
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
