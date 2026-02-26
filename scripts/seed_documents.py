"""Seed medical documents via POST /ingest.

Documents are fetched from Wikipedia REST API (EN + JA) and PubMed E-utilities.
Falls back to bundled sample texts if a network source is unavailable.

Usage:
    python scripts/seed_documents.py [--url URL] [--key API_KEY] [--offline]
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Final

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

# Wikipedia topics: (english_title, japanese_title, filename_basename)
_WIKI_TOPICS: Final[list[tuple[str, str, str]]] = [
    ("Type_2_diabetes", "2型糖尿病", "diabetes_management"),
    ("Hypertension", "高血圧", "hypertension_guidelines"),
    ("Asthma", "喘息", "asthma_management"),
    ("Obesity", "肥満", "obesity_guidelines"),
]

# PubMed queries: (search_term, filename_basename)
_PUBMED_QUERIES: Final[list[tuple[str, str]]] = [
    ("metformin type 2 diabetes treatment guidelines", "metformin_diabetes_pubmed"),
    ("hypertension antihypertensive drug treatment", "hypertension_treatment_pubmed"),
]

_WIKI_API: Final = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
_PUBMED_SEARCH: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_FETCH: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class DocumentFetcher:
    """Fetches medical document text from Wikipedia and PubMed."""

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "healthrag-seeder/1.0 (educational RAG project; github.com/healthrag)"
        )

    def wiki_summary(self, title: str, lang: str) -> str:
        """Return the plain-text extract from a Wikipedia article summary."""
        url = _WIKI_API.format(lang=lang, title=title)
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("extract", "")

    def pubmed_abstract(self, query: str) -> str:
        """Return the plain-text abstract of the top PubMed result for *query*."""
        search = self.session.get(
            _PUBMED_SEARCH,
            params={"db": "pubmed", "term": query, "retmax": 1, "retmode": "json"},
            timeout=10,
        )
        search.raise_for_status()
        ids = search.json()["esearchresult"]["idlist"]
        if not ids:
            return ""
        time.sleep(self.delay)
        fetch = self.session.get(
            _PUBMED_FETCH,
            params={"db": "pubmed", "id": ids[0], "retmode": "text", "rettype": "abstract"},
            timeout=10,
        )
        fetch.raise_for_status()
        return fetch.text.strip()

    def fetch_all(self) -> dict[str, str]:
        """Return {filename: text} for all configured sources."""
        docs: dict[str, str] = {}

        for en_title, ja_title, basename in _WIKI_TOPICS:
            # English Wikipedia
            try:
                text = self.wiki_summary(en_title, "en")
                if text:
                    docs[f"{basename}_en.txt"] = f"{en_title.replace('_', ' ')}\n\n{text}"
                    print(f"  fetched Wikipedia EN: {en_title}")
                else:
                    print(f"  warning Wikipedia EN: {en_title} returned empty extract")
            except Exception as exc:
                print(f"  error Wikipedia EN: {en_title} - {exc}")
            time.sleep(self.delay)

            # Japanese Wikipedia
            try:
                text = self.wiki_summary(ja_title, "ja")
                if text:
                    docs[f"{basename}_ja.txt"] = f"{ja_title}\n\n{text}"
                    print(f"  fetched Wikipedia JA: {ja_title}")
                else:
                    print(f"  warning Wikipedia JA: {ja_title} returned empty extract")
            except Exception as exc:
                print(f"  error Wikipedia JA: {ja_title} - {exc}")
            time.sleep(self.delay)

        for query, basename in _PUBMED_QUERIES:
            try:
                text = self.pubmed_abstract(query)
                if text:
                    docs[f"{basename}.txt"] = text
                    print(f"  fetched PubMed: {query!r}")
                else:
                    print(f"  warning PubMed: {query!r} - no results")
            except Exception as exc:
                print(f"  error PubMed: {query!r} - {exc}")
            time.sleep(self.delay)

        return docs


# Offline fallback used when --offline is set or all network fetches fail
_FALLBACK_DOCUMENTS: Final[dict[str, str]] = {
    "diabetes_management_en.txt": """\
Type 2 Diabetes Management Guidelines

Type 2 diabetes mellitus is a chronic metabolic condition characterised by insulin resistance
and relative insulin deficiency. First-line pharmacotherapy is metformin 500–2000 mg/day,
titrated to minimise gastrointestinal side effects.

HbA1c targets: <7.0% for most adults; <8.0% for elderly patients or those with significant
comorbidities. Monitoring should occur every 3 months until stable, then every 6 months.

Blood pressure management: target <130/80 mmHg. Preferred agents include ACE inhibitors or
ARBs, especially in patients with microalbuminuria. Statin therapy is recommended for
cardiovascular risk reduction in patients over 40 or with existing CVD.

Lifestyle modification: structured diet counselling, at least 150 minutes of moderate-intensity
aerobic exercise per week, and smoking cessation are all Grade A recommendations.

Hypoglycaemia management: mild episodes treated with 15 g fast-acting carbohydrates.
Severe episodes requiring third-party assistance necessitate glucagon administration.
""",
    "hypertension_guidelines_en.txt": """\
Hypertension Clinical Guidelines

Hypertension is defined as sustained systolic BP ≥130 mmHg or diastolic BP ≥80 mmHg.
It affects approximately 1.28 billion adults worldwide and is a leading cause of
cardiovascular disease, stroke, and chronic kidney disease.

Stage 1 Hypertension (130-139/80-89 mmHg): initiate lifestyle modifications.
Stage 2 Hypertension (≥140/90 mmHg): lifestyle modifications plus antihypertensive medication.

First-line agents: thiazide diuretics, calcium channel blockers (CCBs), ACE inhibitors,
or ARBs. Beta-blockers are indicated for patients with ischaemic heart disease or heart failure.

Monitoring: Blood pressure should be checked at every clinical visit. Home BP monitoring
is recommended; target <135/85 mmHg for home readings. Annual laboratory tests include
serum electrolytes, creatinine, eGFR, fasting lipids, and urinalysis.

Resistant hypertension (BP uncontrolled on 3+ agents including a diuretic) requires
specialist referral and evaluation for secondary causes.
""",
    "diabetes_management_ja.txt": """\
2型糖尿病管理ガイドライン

2型糖尿病は、インスリン抵抗性と相対的なインスリン分泌不足を特徴とする慢性代謝疾患です。
第一選択薬はメトホルミン（500〜2000 mg/日）であり、消化器系副作用を最小限にするために
徐々に増量します。

HbA1c目標値：ほとんどの成人では7.0%未満、高齢者または重篤な合併症を有する患者では8.0%未満。
安定するまで3ヶ月ごと、その後は6ヶ月ごとにモニタリングを実施します。

血圧管理：目標値は130/80 mmHg未満。微量アルブミン尿を有する患者にはACE阻害薬またはARBが
推奨されます。40歳以上または既存の心血管疾患を有する患者にはスタチン療法を推奨します。

生活習慣の改善：構造化された食事相談、週150分以上の中等度有酸素運動、禁煙はすべてグレードA推奨です。

低血糖症の管理：軽度のエピソードは15gの速効性炭水化物で治療します。
第三者の援助が必要な重篤なエピソードにはグルカゴン投与が必要です。
""",
    "hypertension_guidelines_ja.txt": """\
高血圧診療ガイドライン

高血圧は、収縮期血圧≥130 mmHgまたは拡張期血圧≥80 mmHgが持続する状態と定義されます。
世界中の約12億8000万人の成人が罹患しており、心血管疾患、脳卒中、慢性腎臓病の主要原因です。

ステージ1高血圧（130-139/80-89 mmHg）：生活習慣の改善を開始します。
ステージ2高血圧（≥140/90 mmHg）：生活習慣の改善に加え、降圧薬の投与を開始します。

第一選択薬：サイアザイド系利尿薬、カルシウム拮抗薬（CCB）、ACE阻害薬、またはARB。
ベータ遮断薬は虚血性心疾患または心不全患者に適応があります。

モニタリング：すべての受診時に血圧測定を行います。家庭血圧モニタリングを推奨し、
目標値は135/85 mmHg未満とします。年1回の検査項目：血清電解質、クレアチニン、eGFR、
空腹時脂質、尿検査。

3剤以上（利尿薬を含む）でもコントロール不良な治療抵抗性高血圧は、専門医への紹介と
二次性高血圧の評価が必要です。
""",
}


def seed(base_url: str, api_key: str, offline: bool = False) -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if offline:
        print("Offline mode — using bundled fallback documents.")
        documents = _FALLBACK_DOCUMENTS
    else:
        print("Fetching documents from Wikipedia and PubMed …")
        fetcher = DocumentFetcher()
        documents = fetcher.fetch_all()
        if not documents:
            print("  All fetches failed — falling back to bundled sample documents.")
            documents = _FALLBACK_DOCUMENTS

    print(f"\nIngesting {len(documents)} documents into {base_url} …")

    for filename, content in documents.items():
        (data_dir / filename).write_text(content, encoding="utf-8")

        resp = requests.post(
            f"{base_url}/ingest",
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
            headers={"X-API-Key": api_key},
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            print(
                f"  ingested {filename} "
                f"[lang={data['language']}, chunks={data['chunks_indexed']}, doc_id={data['doc_id']}]"
            )
        else:
            print(f"  error {filename} - HTTP {resp.status_code}: {resp.text[:200]}")

    print("\nDone. Example queries:")
    print(f"  POST {base_url}/retrieve  {{\"query\": \"Type 2 diabetes treatment\"}}")
    print(f"  POST {base_url}/generate  {{\"query\": \"糖尿病の治療は？\", \"output_language\": \"en\"}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed medical documents into healthrag")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--key", default="dev-key-healthrag", help="API key")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip Wikipedia/PubMed fetching; use bundled fallback documents",
    )
    args = parser.parse_args()
    seed(args.url, args.key, offline=args.offline)
