from core.config import load_settings
from llm.client import create_arc_client


def main() -> None:
    client = create_arc_client(load_settings())

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Explain what an API is in one sentence.",
            }
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
