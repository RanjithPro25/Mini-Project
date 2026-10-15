from pathlib import Path
import json
import re

BASE = Path(__file__).resolve().parent

KB_FILE = (
    BASE
    / "knowledge_base"
    / "snake_knowledge.json"
)

if not KB_FILE.exists():
    raise FileNotFoundError(
        f"Knowledge base not found:\n{KB_FILE}"
    )

with open(
    KB_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError(
        "snake_knowledge.json must contain a list"
    )

print(
    f"Knowledge records loaded: {len(data)}"
)

if len(data) != 313:
    raise ValueError(
        f"Expected 313 records, found {len(data)}"
)


def normalize(text):

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9\s_-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def find_species(query):

    q = normalize(query)

    exact = []

    for item in data:

        scientific = normalize(
            item.get(
                "scientific_name",
                ""
            )
        )

        common = normalize(
            item.get(
                "common_name",
                ""
            )
        )

        if q == scientific or q == common:

            exact.append(item)

    if exact:
        return exact

    partial = []

    for item in data:

        scientific = normalize(
            item.get(
                "scientific_name",
                ""
            )
        )

        common = normalize(
            item.get(
                "common_name",
                ""
            )
        )

        if (
            q in scientific
            or q in common
        ):

            partial.append(item)

    return partial


def show_species(item):

    print()
    print("=" * 65)

    print(
        "Scientific name:",
        item.get(
            "scientific_name",
            "Unknown"
        )
    )

    print(
        "Common name:",
        item.get(
            "common_name",
            "Unknown"
        )
    )

    print(
        "Family:",
        item.get(
            "family",
            "Unknown"
        )
    )

    print(
        "Venomous:",
        item.get(
            "venomous",
            "Unknown"
        )
    )

    print(
        "Venomous scale:",
        item.get(
            "venomous_scale",
            "Unknown"
        )
    )

    print()

    print(
        "Identification:"
    )

    print(
        item.get(
            "identification_features",
            "Not available"
        )
    )

    print()

    print(
        "Habitat:"
    )

    print(
        item.get(
            "habitat",
            "Not available"
        )
    )

    print()

    print(
        "Behavior:"
    )

    print(
        item.get(
            "behavior",
            "Not available"
        )
    )

    print()

    print(
        "Precautions:"
    )

    print(
        item.get(
            "specific_precautions",
            "Not available"
        )
    )

    print()

    print(
        "What NOT to do:"
    )

    print(
        item.get(
            "what_not_to_do",
            "Not available"
        )
    )

    print()

    print(
        "Snakebite first aid:"
    )

    print(
        item.get(
            "snakebite_first_aid",
            "Not available"
        )
    )

    print()

    print(
        "Emergency level:",
        item.get(
            "emergency_level",
            "Unknown"
        )
    )

    print()

    print(
        "Medical attention:"
    )

    print(
        item.get(
            "medical_attention",
            "Not available"
        )
    )

    print()

    print(
        "Source:",
        item.get(
            "source",
            "Not available"
        )
    )

    print("=" * 65)


def answer(query):

    q = normalize(query)

    if not q:
        return

    results = find_species(q)

    if results:

        if len(results) == 1:

            show_species(
                results[0]
            )

        else:

            print()
            print(
                "Multiple species found:"
            )

            for i, item in enumerate(
                results[:10],
                1
            ):

                print(
                    f"{i}. "
                    f"{item.get('scientific_name', '')}"
                    f" - "
                    f"{item.get('common_name', '')}"
                )

            print()

            choice = input(
                "Enter number: "
            ).strip()

            if choice.isdigit():

                index = int(choice) - 1

                if (
                    0 <= index
                    < len(results)
                ):

                    show_species(
                        results[index]
                    )

                else:

                    print(
                        "Invalid selection."
                    )

    else:

        print()
        print(
            "Species not found in "
            "VenomGuard knowledge base."
        )

        print(
            "Try the scientific or common name."
        )


print()
print("=" * 65)
print(
    "VENOMGUARD PHASE 15"
)
print(
    "TEXT KNOWLEDGE ENGINE"
)
print("=" * 65)

print()
print(
    "Type a snake scientific/common name."
)

print(
    "Type 'exit' to close."
)

print()

while True:

    query = input(
        "VenomGuard > "
    ).strip()

    if query.lower() in [
        "exit",
        "quit"
    ]:

        break

    answer(query)

print()
print(
    "PHASE 15 TEXT ENGINE FINISHED SUCCESSFULLY"
)