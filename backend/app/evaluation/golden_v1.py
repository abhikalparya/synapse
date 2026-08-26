"""Source for learning_graph_eval_v1. Run as ``python -m app.evaluation.golden_v1`` to emit JSONL."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.dataset import write_dataset
from app.evaluation.schemas import EvalExample

_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = _REPO_ROOT / "data" / "eval" / "learning_graph_eval_v1.jsonl"


def _e(
    id: str,
    category: str,
    difficulty: str,
    goal: str,
    topics: list[str],
    deps: list[tuple[str, str]],
    *,
    notes: str,
    input_notes: str | None = None,
    aliases: dict[str, list[str]] | None = None,
    extras: list[str] | None = None,
    summaries: dict[str, str] | None = None,
) -> EvalExample:
    return EvalExample(
        id=id,
        category=category,
        difficulty=difficulty,  # type: ignore[arg-type]
        goal=goal,
        gold_topics=topics,
        gold_dependencies=deps,
        input_notes=input_notes,
        notes=notes,
        topic_aliases=aliases or {},
        allowed_extra_topics=extras or [],
        gold_topic_summaries=summaries or {},
    )


def golden_v1() -> list[EvalExample]:
    """40 gold graphs. Edges are Synapse-directed: from requires to."""
    examples = [
        _e(
            "python_basics_001",
            "programming",
            "beginner",
            "Learn the basics of Python programming",
            ["Variables", "Control Flow", "Functions", "Data Structures", "Python Basics"],
            [
                ("Control Flow", "Variables"),
                ("Functions", "Control Flow"),
                ("Data Structures", "Functions"),
                ("Python Basics", "Data Structures"),
            ],
            notes="A first-language path: bind values, branch, abstract, then collect.",
        ),
        _e(
            "git_collaboration_001",
            "programming",
            "intermediate",
            "Learn collaborative Git workflows used on software teams",
            ["Command Line", "Git Basics", "Branching", "Pull Requests", "Code Review"],
            [
                ("Git Basics", "Command Line"),
                ("Branching", "Git Basics"),
                ("Pull Requests", "Branching"),
                ("Code Review", "Pull Requests"),
            ],
            notes="PRs are meaningless without branches; review sits on top of PRs.",
        ),
        _e(
            "compilers_001",
            "programming",
            "advanced",
            "Learn how compilers turn source code into machine code",
            [
                "Regular Languages",
                "Context-Free Grammars",
                "Parsing",
                "Semantic Analysis",
                "Code Generation",
                "Compilers",
            ],
            [
                ("Parsing", "Regular Languages"),
                ("Parsing", "Context-Free Grammars"),
                ("Semantic Analysis", "Parsing"),
                ("Code Generation", "Semantic Analysis"),
                ("Compilers", "Code Generation"),
            ],
            notes="Frontend analysis before backend generation; both language-theory roots feed parsing.",
        ),
        _e(
            "rust_ownership_001",
            "programming",
            "intermediate",
            "Understand Rust ownership, borrowing, and lifetimes",
            ["Stack and Heap", "Ownership", "Borrowing", "Lifetimes", "Rust Memory Safety"],
            [
                ("Ownership", "Stack and Heap"),
                ("Borrowing", "Ownership"),
                ("Lifetimes", "Borrowing"),
                ("Rust Memory Safety", "Lifetimes"),
            ],
            notes="Safety is the goal; lifetimes only make sense after borrowing.",
        ),
        _e(
            "supervised_learning_001",
            "machine_learning",
            "beginner",
            "Learn supervised machine learning for tabular prediction",
            ["Data Splits", "Linear Regression", "Loss Functions", "Overfitting", "Supervised Learning"],
            [
                ("Linear Regression", "Data Splits"),
                ("Loss Functions", "Linear Regression"),
                ("Overfitting", "Loss Functions"),
                ("Supervised Learning", "Overfitting"),
            ],
            notes="Holdout comes first so later fitting talk isn't circular.",
        ),
        _e(
            "transformers_001",
            "machine_learning",
            "intermediate",
            "Learn transformer-based language models",
            [
                "Linear Algebra",
                "Probability",
                "Neural Networks",
                "Attention Mechanisms",
                "Transformers",
                "Large Language Models",
            ],
            [
                ("Neural Networks", "Linear Algebra"),
                ("Neural Networks", "Probability"),
                ("Attention Mechanisms", "Neural Networks"),
                ("Transformers", "Attention Mechanisms"),
                ("Large Language Models", "Transformers"),
            ],
            notes="Standard LLM stack; math is shared foundation for networks, not a sibling of Transformers.",
            aliases={"Large Language Models": ["LLMs", "LLM", "Large Language Model"]},
            extras=["Tokenization"],
        ),
        _e(
            "rlhf_001",
            "machine_learning",
            "advanced",
            "Learn reinforcement learning from human feedback for language models",
            [
                "Language Models",
                "Supervised Fine-Tuning",
                "Reward Models",
                "Policy Optimization",
                "RLHF",
            ],
            [
                ("Supervised Fine-Tuning", "Language Models"),
                ("Reward Models", "Supervised Fine-Tuning"),
                ("Policy Optimization", "Reward Models"),
                ("RLHF", "Policy Optimization"),
            ],
            notes="SFT then preference reward then policy update; RLHF names the full loop.",
        ),
        _e(
            "cnn_vision_001",
            "machine_learning",
            "intermediate",
            "Learn convolutional neural networks for image classification",
            ["Linear Algebra", "Convolutions", "CNNs", "Pooling", "Image Classification"],
            [
                ("Convolutions", "Linear Algebra"),
                ("CNNs", "Convolutions"),
                ("Pooling", "CNNs"),
                ("Image Classification", "Pooling"),
            ],
            notes="Classification is the task head after spatial hierarchy is in place.",
        ),
        _e(
            "calculus_limits_001",
            "mathematics",
            "beginner",
            "Learn introductory differential calculus",
            ["Functions", "Limits", "Continuity", "Derivatives", "Differential Calculus"],
            [
                ("Limits", "Functions"),
                ("Continuity", "Limits"),
                ("Derivatives", "Continuity"),
                ("Differential Calculus", "Derivatives"),
            ],
            notes="Textbook order; derivatives are defined via limits.",
        ),
        _e(
            "linear_algebra_001",
            "mathematics",
            "intermediate",
            "Learn core linear algebra used in applied science",
            ["Vectors", "Matrices", "Linear Transformations", "Eigenvalues", "Linear Algebra"],
            [
                ("Matrices", "Vectors"),
                ("Linear Transformations", "Matrices"),
                ("Eigenvalues", "Linear Transformations"),
                ("Linear Algebra", "Eigenvalues"),
            ],
            notes="Maps before spectral theory.",
        ),
        _e(
            "real_analysis_001",
            "mathematics",
            "advanced",
            "Learn the foundations of real analysis",
            ["Sequences", "Series", "Metric Spaces", "Continuity", "Differentiation", "Real Analysis"],
            [
                ("Series", "Sequences"),
                ("Metric Spaces", "Sequences"),
                ("Continuity", "Metric Spaces"),
                ("Differentiation", "Continuity"),
                ("Real Analysis", "Series"),
                ("Real Analysis", "Differentiation"),
            ],
            notes="Two branches (series vs metric continuity) join at the subject.",
        ),
        _e(
            "probability_001",
            "mathematics",
            "intermediate",
            "Learn undergraduate probability",
            ["Combinatorics", "Random Variables", "Expectation", "Conditional Probability", "Probability"],
            [
                ("Random Variables", "Combinatorics"),
                ("Expectation", "Random Variables"),
                ("Conditional Probability", "Random Variables"),
                ("Probability", "Expectation"),
                ("Probability", "Conditional Probability"),
            ],
            notes="Conditioning and expectation are parallel uses of random variables.",
        ),
        _e(
            "sql_basics_001",
            "databases",
            "beginner",
            "Learn to query relational data with SQL",
            ["Relational Model", "Tables", "SELECT Queries", "Joins", "SQL"],
            [
                ("Tables", "Relational Model"),
                ("SELECT Queries", "Tables"),
                ("Joins", "SELECT Queries"),
                ("SQL", "Joins"),
            ],
            notes="Joins after single-table SELECT.",
        ),
        _e(
            "indexing_001",
            "databases",
            "intermediate",
            "Learn how database indexes speed up queries",
            ["Disk Storage", "B-Trees", "Indexes", "Query Planning", "Database Indexing"],
            [
                ("B-Trees", "Disk Storage"),
                ("Indexes", "B-Trees"),
                ("Query Planning", "Indexes"),
                ("Database Indexing", "Query Planning"),
            ],
            notes="Physical layout explains why B-trees exist.",
        ),
        _e(
            "distributed_db_001",
            "databases",
            "advanced",
            "Learn distributed database design",
            ["Replication", "Partitioning", "Consensus", "Consistency Models", "Distributed Databases"],
            [
                ("Consensus", "Replication"),
                ("Consistency Models", "Replication"),
                ("Consistency Models", "Partitioning"),
                ("Distributed Databases", "Consensus"),
                ("Distributed Databases", "Consistency Models"),
            ],
            notes="Consensus is one tool; consistency models also depend on how data is split.",
        ),
        _e(
            "normalization_001",
            "databases",
            "intermediate",
            "Learn relational schema normalization",
            ["Relations", "Functional Dependencies", "Normal Forms", "Normalization", "Schema Design"],
            [
                ("Functional Dependencies", "Relations"),
                ("Normal Forms", "Functional Dependencies"),
                ("Normalization", "Normal Forms"),
                ("Schema Design", "Normalization"),
            ],
            notes="NF definitions before the process of normalizing a schema.",
        ),
        _e(
            "client_server_001",
            "distributed_systems",
            "beginner",
            "Learn how client-server systems communicate over the network",
            ["Networking Basics", "HTTP", "Client-Server", "APIs", "Distributed Systems Intro"],
            [
                ("HTTP", "Networking Basics"),
                ("Client-Server", "HTTP"),
                ("APIs", "Client-Server"),
                ("Distributed Systems Intro", "APIs"),
            ],
            notes="HTTP is the first distributed protocol most learners meet.",
        ),
        _e(
            "consensus_001",
            "distributed_systems",
            "intermediate",
            "Learn consensus algorithms for replicated state",
            ["Failure Models", "Replication", "Paxos", "Raft", "Consensus"],
            [
                ("Replication", "Failure Models"),
                ("Paxos", "Replication"),
                ("Raft", "Replication"),
                ("Consensus", "Paxos"),
                ("Consensus", "Raft"),
            ],
            notes="Paxos and Raft are sibling algorithms, not a chain.",
        ),
        _e(
            "cap_consistency_001",
            "distributed_systems",
            "advanced",
            "Understand consistency trade-offs in distributed systems",
            ["Replication", "CAP Theorem", "Eventual Consistency", "Linearizability", "Consistency Models"],
            [
                ("CAP Theorem", "Replication"),
                ("Eventual Consistency", "CAP Theorem"),
                ("Linearizability", "CAP Theorem"),
                ("Consistency Models", "Eventual Consistency"),
                ("Consistency Models", "Linearizability"),
            ],
            notes="CAP frames the two example points on the consistency spectrum.",
        ),
        _e(
            "queues_001",
            "distributed_systems",
            "intermediate",
            "Learn asynchronous processing with message queues",
            ["Producer-Consumer", "Message Queues", "Delivery Guarantees", "Backpressure", "Async Processing"],
            [
                ("Message Queues", "Producer-Consumer"),
                ("Delivery Guarantees", "Message Queues"),
                ("Backpressure", "Message Queues"),
                ("Async Processing", "Delivery Guarantees"),
                ("Async Processing", "Backpressure"),
            ],
            notes="Guarantees and backpressure are independent queue properties.",
        ),
        _e(
            "cloud_intro_001",
            "cloud_computing",
            "beginner",
            "Learn the basic ideas of cloud computing",
            ["Virtualization", "IaaS", "PaaS", "SaaS", "Cloud Computing"],
            [
                ("IaaS", "Virtualization"),
                ("PaaS", "IaaS"),
                ("SaaS", "PaaS"),
                ("Cloud Computing", "SaaS"),
            ],
            notes="Service models stacked from more control to less.",
            aliases={"IaaS": ["Infrastructure as a Service"], "PaaS": ["Platform as a Service"], "SaaS": ["Software as a Service"]},
        ),
        _e(
            "containers_k8s_001",
            "cloud_computing",
            "intermediate",
            "Learn to run applications on Kubernetes",
            ["Containers", "Docker", "Orchestration", "Kubernetes", "Deployments"],
            [
                ("Docker", "Containers"),
                ("Orchestration", "Docker"),
                ("Kubernetes", "Orchestration"),
                ("Deployments", "Kubernetes"),
            ],
            notes="k8s is one orchestrator; Deployments are a k8s primitive.",
        ),
        _e(
            "sre_reliability_001",
            "cloud_computing",
            "advanced",
            "Learn site reliability engineering practices",
            ["SLIs and SLOs", "Error Budgets", "Incident Response", "Capacity Planning", "SRE"],
            [
                ("Error Budgets", "SLIs and SLOs"),
                ("Incident Response", "Error Budgets"),
                ("Capacity Planning", "SLIs and SLOs"),
                ("SRE", "Incident Response"),
                ("SRE", "Capacity Planning"),
            ],
            notes="Error budgets come from SLOs; ops work splits incidents vs capacity.",
        ),
        _e(
            "iam_cloud_001",
            "cloud_computing",
            "intermediate",
            "Learn identity and access management in the cloud",
            ["Identities", "Authentication", "Authorization", "IAM Policies", "Cloud IAM"],
            [
                ("Authentication", "Identities"),
                ("Authorization", "Authentication"),
                ("IAM Policies", "Authorization"),
                ("Cloud IAM", "IAM Policies"),
            ],
            notes="Authn before authz; policies encode authz.",
        ),
        _e(
            "html_css_001",
            "frontend_engineering",
            "beginner",
            "Learn to build a static web page with HTML and CSS",
            ["HTML", "CSS Selectors", "Layout", "Responsive Design", "Web Pages"],
            [
                ("CSS Selectors", "HTML"),
                ("Layout", "CSS Selectors"),
                ("Responsive Design", "Layout"),
                ("Web Pages", "Responsive Design"),
            ],
            notes="Structure before style before layout before breakpoints.",
        ),
        _e(
            "react_state_001",
            "frontend_engineering",
            "intermediate",
            "Learn React component state",
            ["JavaScript", "DOM", "Components", "State", "React"],
            [
                ("DOM", "JavaScript"),
                ("Components", "DOM"),
                ("State", "Components"),
                ("React", "State"),
            ],
            notes="React is the framework goal; state is the key mechanic.",
        ),
        _e(
            "web_perf_001",
            "frontend_engineering",
            "advanced",
            "Learn web performance optimization",
            ["Critical Rendering Path", "Bundling", "Caching", "Core Web Vitals", "Web Performance"],
            [
                ("Bundling", "Critical Rendering Path"),
                ("Caching", "Critical Rendering Path"),
                ("Core Web Vitals", "Bundling"),
                ("Core Web Vitals", "Caching"),
                ("Web Performance", "Core Web Vitals"),
            ],
            notes="CRP explains why bundling and caching matter; vitals measure the result.",
        ),
        _e(
            "accessibility_001",
            "frontend_engineering",
            "intermediate",
            "Learn to make web interfaces accessible",
            ["Semantic HTML", "Keyboard Navigation", "ARIA", "Screen Readers", "Accessibility"],
            [
                ("Keyboard Navigation", "Semantic HTML"),
                ("ARIA", "Semantic HTML"),
                ("Screen Readers", "ARIA"),
                ("Accessibility", "Keyboard Navigation"),
                ("Accessibility", "Screen Readers"),
            ],
            notes="ARIA supplements semantics; screen readers consume both.",
        ),
        _e(
            "rest_api_001",
            "backend_engineering",
            "beginner",
            "Learn to design a REST API",
            ["HTTP", "Resources", "REST", "Status Codes", "API Design"],
            [
                ("Resources", "HTTP"),
                ("REST", "Resources"),
                ("Status Codes", "HTTP"),
                ("API Design", "REST"),
                ("API Design", "Status Codes"),
            ],
            notes="REST uses resources; status codes are HTTP and apply to any API design.",
        ),
        _e(
            "authn_authz_001",
            "backend_engineering",
            "intermediate",
            "Learn authentication and authorization for backend services",
            ["Sessions", "Tokens", "Authentication", "Authorization", "Backend Auth"],
            [
                ("Authentication", "Sessions"),
                ("Authentication", "Tokens"),
                ("Authorization", "Authentication"),
                ("Backend Auth", "Authorization"),
            ],
            notes="Sessions and tokens are sibling authn mechanisms.",
        ),
        _e(
            "distributed_tx_001",
            "backend_engineering",
            "advanced",
            "Learn distributed transaction patterns",
            ["ACID", "Two-Phase Commit", "Sagas", "Outbox Pattern", "Distributed Transactions"],
            [
                ("Two-Phase Commit", "ACID"),
                ("Sagas", "ACID"),
                ("Outbox Pattern", "Sagas"),
                ("Distributed Transactions", "Two-Phase Commit"),
                ("Distributed Transactions", "Outbox Pattern"),
            ],
            notes="2PC vs saga+outbox as alternative implementations of cross-service atomicity.",
        ),
        _e(
            "caching_001",
            "backend_engineering",
            "intermediate",
            "Learn caching strategies for backend systems",
            ["Latency", "Cache Invalidation", "Redis", "CDN", "Caching Strategies"],
            [
                ("Cache Invalidation", "Latency"),
                ("Redis", "Cache Invalidation"),
                ("CDN", "Cache Invalidation"),
                ("Caching Strategies", "Redis"),
                ("Caching Strategies", "CDN"),
            ],
            notes="Invalidation is the hard problem; Redis vs CDN are placement choices.",
        ),
        _e(
            "etl_intro_001",
            "data_engineering",
            "beginner",
            "Learn the ETL pipeline pattern",
            ["Source Systems", "Extraction", "Transformation", "Loading", "ETL"],
            [
                ("Extraction", "Source Systems"),
                ("Transformation", "Extraction"),
                ("Loading", "Transformation"),
                ("ETL", "Loading"),
            ],
            notes="Literal E-T-L order.",
        ),
        _e(
            "warehousing_001",
            "data_engineering",
            "intermediate",
            "Learn dimensional modeling for a data warehouse",
            ["Dimensional Modeling", "Star Schema", "Warehouses", "Batch Pipelines", "Data Warehousing"],
            [
                ("Star Schema", "Dimensional Modeling"),
                ("Warehouses", "Star Schema"),
                ("Batch Pipelines", "Warehouses"),
                ("Data Warehousing", "Batch Pipelines"),
            ],
            notes="Model then store then fill.",
        ),
        _e(
            "stream_processing_001",
            "data_engineering",
            "advanced",
            "Learn stream processing architectures",
            ["Event Logs", "Kafka", "Windowing", "Exactly-Once", "Stream Processing"],
            [
                ("Kafka", "Event Logs"),
                ("Windowing", "Kafka"),
                ("Exactly-Once", "Kafka"),
                ("Stream Processing", "Windowing"),
                ("Stream Processing", "Exactly-Once"),
            ],
            notes="Windowing and delivery semantics are independent Kafka-consuming concerns.",
        ),
        _e(
            "data_quality_001",
            "data_engineering",
            "intermediate",
            "Learn how to test data quality in pipelines",
            ["Schemas", "Validation", "Freshness", "Data Tests", "Data Quality"],
            [
                ("Validation", "Schemas"),
                ("Data Tests", "Validation"),
                ("Data Tests", "Freshness"),
                ("Data Quality", "Data Tests"),
            ],
            notes="Freshness is a quality dimension that does not require schemas.",
        ),
        _e(
            "security_basics_001",
            "security",
            "beginner",
            "Learn security fundamentals for software developers",
            ["Threats", "Authentication", "Least Privilege", "Encryption Basics", "Security Fundamentals"],
            [
                ("Authentication", "Threats"),
                ("Least Privilege", "Threats"),
                ("Encryption Basics", "Threats"),
                ("Security Fundamentals", "Authentication"),
                ("Security Fundamentals", "Least Privilege"),
                ("Security Fundamentals", "Encryption Basics"),
            ],
            notes="Three parallel controls against a threat model.",
        ),
        _e(
            "web_security_001",
            "security",
            "intermediate",
            "Learn common web application vulnerabilities",
            ["HTTP", "Sessions", "XSS", "CSRF", "SQL Injection", "Web Security"],
            [
                ("Sessions", "HTTP"),
                ("XSS", "HTTP"),
                ("CSRF", "Sessions"),
                ("SQL Injection", "HTTP"),
                ("Web Security", "XSS"),
                ("Web Security", "CSRF"),
                ("Web Security", "SQL Injection"),
            ],
            notes="CSRF needs session/cookie context; XSS and SQLi are HTTP-surface bugs.",
            input_notes=(
                "I keep seeing XSS, CSRF, and SQL injection in bug bounties. "
                "I know HTTP and cookies exist but I do not know how the attacks relate."
            ),
        ),
        _e(
            "crypto_applied_001",
            "security",
            "advanced",
            "Learn applied cryptography used on the web",
            ["Symmetric Crypto", "Public Key Crypto", "Hashing", "TLS", "Applied Cryptography"],
            [
                ("TLS", "Symmetric Crypto"),
                ("TLS", "Public Key Crypto"),
                ("TLS", "Hashing"),
                ("Applied Cryptography", "TLS"),
            ],
            notes="TLS is the motivating composition of the three primitives.",
        ),
        _e(
            "appsec_sdlc_001",
            "security",
            "intermediate",
            "Learn application security in the software development lifecycle",
            ["Threat Modeling", "Code Review", "SAST", "Dependency Scanning", "Secure SDLC"],
            [
                ("Code Review", "Threat Modeling"),
                ("SAST", "Code Review"),
                ("Dependency Scanning", "Threat Modeling"),
                ("Secure SDLC", "SAST"),
                ("Secure SDLC", "Dependency Scanning"),
            ],
            notes="Threat model first; SAST sits on reviewed code, deps are a parallel supply-chain track.",
        ),
    ]
    _apply_quality_annotations(examples)
    return examples


def _aliases(example: EvalExample, mapping: dict[str, list[str]]) -> None:
    for gold, names in mapping.items():
        example.topic_aliases.setdefault(gold, [])
        for a in names:
            if a not in example.topic_aliases[gold]:
                example.topic_aliases[gold].append(a)


def _apply_quality_annotations(examples: list[EvalExample]) -> None:
    """Curated aliases / optional topics / acceptable edges for the worst quality cases.

    Gold ``gold_topics`` / ``gold_dependencies`` stay intact for backward compatibility.
    Aliases are hand-written from observed generations — not LLM-invented.
    """
    by_id = {e.id: e for e in examples}
    limitation = (
        " The benchmark measures agreement with curated reference structures and does not "
        "claim that there is only one universally correct learning graph."
    )

    e = by_id["python_basics_001"]
    e.required_topics = ["Variables", "Control Flow", "Functions", "Data Structures"]
    e.optional_topics = ["File I/O", "Python Syntax"]
    e.required_dependencies = [
        ("Control Flow", "Variables"),
        ("Functions", "Control Flow"),
        ("Data Structures", "Functions"),
    ]
    _aliases(e, {"Control Flow": ["Control Structures"]})
    e.acceptable_dependencies = [("Data Structures", "Control Flow"), ("Functions", "Variables")]
    e.notes += limitation

    e = by_id["sql_basics_001"]
    e.required_topics = ["Tables", "SELECT Queries", "Joins"]
    e.optional_topics = ["Filtering Data", "Sorting Results", "Basic SQL Syntax"]
    e.required_dependencies = [("SELECT Queries", "Tables"), ("Joins", "SELECT Queries")]
    _aliases(
        e,
        {
            "SELECT Queries": ["SELECT Statement", "SELECT"],
            "Joins": ["Joining Tables", "Table Joins"],
            "SQL": ["Introduction to SQL"],
        },
    )
    e.acceptable_dependencies = [("Joins", "Tables")]
    e.notes += limitation

    e = by_id["transformers_001"]
    e.required_topics = ["Neural Networks", "Attention Mechanisms", "Transformers"]
    e.optional_topics = ["Tokenization", "Training Transformer Models", "Introduction to Language Models"]
    e.required_dependencies = [
        ("Attention Mechanisms", "Neural Networks"),
        ("Transformers", "Attention Mechanisms"),
    ]
    _aliases(
        e,
        {
            "Neural Networks": ["Basics of Neural Networks"],
            "Transformers": ["Understanding Transformers", "Transformer Architecture"],
        },
    )
    e.acceptable_dependencies = [("Transformers", "Neural Networks"), ("Large Language Models", "Neural Networks")]
    e.notes += limitation

    e = by_id["cnn_vision_001"]
    e.required_topics = ["Convolutions", "CNNs", "Image Classification"]
    e.optional_topics = ["Image Preprocessing Techniques", "Introduction to Neural Networks", "Neural Networks"]
    e.required_dependencies = [("CNNs", "Convolutions"), ("Image Classification", "CNNs")]
    _aliases(e, {"CNNs": ["Convolutional Neural Networks", "Convolutional Neural Networks (CNNs)"]})
    e.notes += limitation

    e = by_id["crypto_applied_001"]
    e.required_topics = ["Symmetric Crypto", "Public Key Crypto", "Hashing"]
    e.optional_topics = ["Digital Signatures", "Introduction to Cryptography"]
    e.required_dependencies = []
    _aliases(
        e,
        {
            "Symmetric Crypto": ["Symmetric Encryption"],
            "Public Key Crypto": ["Asymmetric Encryption", "Public Key Encryption"],
            "Hashing": ["Hash Functions"],
            "TLS": ["Transport Layer Security", "HTTPS"],
        },
    )
    e.acceptable_dependencies = [("TLS", "Symmetric Crypto")]
    e.notes += limitation

    e = by_id["html_css_001"]
    e.required_topics = ["HTML", "Layout", "Responsive Design"]
    e.optional_topics = ["HTML Document Structure", "CSS Fundamentals", "CSS Selectors"]
    e.required_dependencies = [("Layout", "HTML"), ("Responsive Design", "Layout")]
    _aliases(
        e,
        {
            "HTML": ["HTML Basics"],
            "Layout": ["CSS Layout Techniques", "CSS Layout"],
            "Responsive Design": ["Responsive Design Principles"],
            "Web Pages": ["Building a Static Web Page", "Static Web Page"],
        },
    )
    e.notes += limitation

    e = by_id["git_collaboration_001"]
    e.required_topics = ["Git Basics", "Branching"]
    e.optional_topics = ["Resolving Merge Conflicts", "Collaborative Workflows", "Command Line"]
    e.required_dependencies = [("Branching", "Git Basics")]
    _aliases(
        e,
        {
            "Git Basics": ["Introduction to Git", "Basic Git Commands"],
            "Branching": ["Understanding Branching"],
        },
    )
    e.acceptable_dependencies = [("Pull Requests", "Git Basics")]
    e.notes += limitation

    e = by_id["containers_k8s_001"]
    e.required_topics = ["Containers", "Kubernetes"]
    e.optional_topics = ["Kubernetes Networking", "Kubernetes Storage", "Docker", "Orchestration"]
    e.required_dependencies = [("Kubernetes", "Containers")]
    _aliases(
        e,
        {
            "Containers": ["Containerization"],
            "Kubernetes": ["Kubernetes Basics"],
            "Deployments": ["Kubernetes Deployment"],
        },
    )
    e.notes += limitation

    e = by_id["etl_intro_001"]
    e.required_topics = ["Extraction", "Transformation", "Loading"]
    e.optional_topics = ["ETL Tools"]
    e.required_dependencies = [
        ("Transformation", "Extraction"),
        ("Loading", "Transformation"),
    ]
    _aliases(
        e,
        {
            "Extraction": ["Data Extraction"],
            "Transformation": ["Data Transformation"],
            "Loading": ["Data Loading"],
            "ETL": ["ETL Overview"],
        },
    )
    e.acceptable_dependencies = [("Loading", "Extraction")]
    e.notes += limitation

    e = by_id["authn_authz_001"]
    e.required_topics = ["Authentication", "Authorization"]
    e.optional_topics = ["Implementing Authentication", "Implementing Authorization"]
    e.required_dependencies = [("Authorization", "Authentication")]
    _aliases(
        e,
        {
            "Authentication": ["Introduction to Authentication", "Authentication Methods"],
            "Authorization": ["Introduction to Authorization", "Authorization Models"],
        },
    )
    e.acceptable_dependencies = [("Authorization", "Sessions"), ("Authorization", "Tokens")]
    e.notes += limitation

    e = by_id["calculus_limits_001"]
    e.required_topics = ["Functions", "Limits", "Derivatives"]
    e.optional_topics = ["Applications of Derivatives"]
    e.required_dependencies = [("Limits", "Functions"), ("Derivatives", "Limits")]
    e.acceptable_dependencies = [("Derivatives", "Limits")]
    e.notes += limitation

    e = by_id["rlhf_001"]
    e.required_topics = ["Language Models", "RLHF"]
    e.optional_topics = ["Reinforcement Learning Basics", "Human Feedback in Machine Learning"]
    e.required_dependencies = [("RLHF", "Language Models")]
    _aliases(
        e,
        {
            "Language Models": ["Language Models Overview"],
            "RLHF": ["Reinforcement Learning from Human Feedback"],
        },
    )
    e.notes += limitation

    e = by_id["supervised_learning_001"]
    e.required_topics = ["Supervised Learning", "Overfitting"]
    e.optional_topics = ["Data Preprocessing Techniques", "Tabular Data", "Understanding Tabular Data"]
    e.required_dependencies = [("Supervised Learning", "Overfitting")]
    _aliases(e, {"Supervised Learning": ["Introduction to Supervised Learning"]})
    e.notes += limitation

    e = by_id["normalization_001"]
    e.required_topics = ["Functional Dependencies", "Normalization"]
    e.optional_topics = ["First Normal Form (1NF)", "Second Normal Form (2NF)", "Third Normal Form (3NF)"]
    e.required_dependencies = [("Normalization", "Functional Dependencies")]
    _aliases(
        e,
        {
            "Relations": ["Relational Database Basics"],
            "Normalization": ["Normalization Process"],
            "Normal Forms": ["First Normal Form", "1NF"],
        },
    )
    e.notes += limitation

    e = by_id["appsec_sdlc_001"]
    e.required_topics = ["Threat Modeling", "Secure SDLC"]
    e.optional_topics = [
        "Security in Design Phase",
        "Security in Implementation Phase",
        "Security Testing",
        "Application Security Fundamentals",
    ]
    e.required_dependencies = [("Secure SDLC", "Threat Modeling")]
    _aliases(e, {"Secure SDLC": ["Software Development Lifecycle"]})
    e.notes += limitation


def main() -> None:
    examples = golden_v1()
    _apply_quality_annotations(examples)
    for ex in examples:
        ex.dataset_version = "learning_graph_eval_v1"
    write_dataset(examples, OUTPUT)
    quality_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    for ex in examples:
        ex.dataset_version = "learning_graph_quality_v1"
    write_dataset(examples, quality_path)
    print(f"Wrote {len(examples)} examples to {OUTPUT}")
    print(f"Wrote {len(examples)} examples to {quality_path}")


if __name__ == "__main__":
    main()
