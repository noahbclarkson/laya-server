//! Checks laya-server end to end through the `typesafe-api` crate.
//!
//! Start the server (natively or with `docker compose up -d`), then from this
//! directory:
//!
//! ```sh
//! cargo test -- --nocapture
//! ```
//!
//! `LAYA_SERVER_URL` overrides the address (default `http://127.0.0.1:8765`),
//! and `LAYA_SERVER_API_KEY` supplies the token when the server requires one.
//!
//! The assertions are on shape, not on what the model concludes. The base
//! checkpoints are weak zero-shot, so asserting a verdict would test Laya
//! rather than the server.

use std::time::Duration;

use typesafe_api::{
    ApiErrorKind, ChoiceOf, Client, ClientBuilder, Error, Evaluation, JEV_LATEST, Levels, Noul,
    NoulAnswer, Options, RetryPolicy, ScoreOf, questions,
};

const DEFAULT_URL: &str = "http://127.0.0.1:8765";

#[derive(Options, Debug, PartialEq)]
enum DocumentKind {
    /// A balance sheet, profit and loss statement, cash flow statement, or a full set of annual accounts
    FinancialStatements,
    /// A bill from a supplier asking for payment
    Invoice,
    /// A list of transactions sent by a bank
    BankStatement,
    /// Any other document
    Other,
}

#[derive(Levels, Debug, PartialEq)]
enum Completeness {
    /// A fragment; most of the figures are missing
    Fragment,
    /// Readable, but sections or comparatives are missing
    Partial,
    /// Complete, with every section and comparative present
    Complete,
}

#[derive(Evaluation, Debug)]
struct Classification {
    /// Is this document a set of financial statements?
    is_financial_statements: NoulAnswer,
    /// What kind of document is this?
    kind: ChoiceOf<DocumentKind>,
    /// How complete is the document?
    completeness: ScoreOf<Completeness>,
}

const STATEMENT: &str = "ACME TRADING LIMITED\n\
     STATEMENT OF FINANCIAL POSITION AS AT 31 MARCH 2026\n\
     Current assets: cash 42,100; trade receivables 18,450; inventory 9,300.\n\
     Non-current assets: plant and equipment 120,000.\n\
     Current liabilities: trade payables 15,200; GST payable 3,100.\n\
     Equity: share capital 100,000; retained earnings 71,550.";

/// A client for the server under test, with the crate's default model.
fn builder() -> ClientBuilder {
    Client::builder()
        .base_url(std::env::var("LAYA_SERVER_URL").unwrap_or_else(|_| DEFAULT_URL.to_owned()))
        .api_key(std::env::var("LAYA_SERVER_API_KEY").unwrap_or_else(|_| "local".to_owned()))
        // Generous, because a CPU takes seconds where a GPU takes milliseconds.
        .timeout(Duration::from_mins(1))
        .retry(RetryPolicy::none())
}

fn client() -> Client {
    builder().model("laya").build().expect("the client builds")
}

/// Unwraps a call, turning "nothing is listening" into the fix for it.
fn reached<T>(result: Result<T, Error>) -> T {
    match result {
        Ok(value) => value,
        Err(Error::Transport { url, source }) => panic!(
            "could not reach laya-server at {url} ({source}); start it with \
             `python server.py` or `docker compose up -d`, or set LAYA_SERVER_URL"
        ),
        Err(error) => panic!("{error}"),
    }
}

fn assert_distribution<'a>(what: &str, probabilities: impl IntoIterator<Item = &'a f64>) {
    let total: f64 = probabilities.into_iter().sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "{what}: probabilities should sum to about 1, got {total}"
    );
}

#[tokio::test]
async fn the_model_list_names_the_router_and_a_checkpoint() {
    let models = reached(client().models().await);
    println!("{models:#?}");

    assert!(
        models.iter().any(|model| model.name == "laya"),
        "{models:#?}"
    );
    assert!(
        models.iter().any(|model| model.name.starts_with("laya-")),
        "{models:#?}"
    );
}

#[tokio::test]
async fn a_typed_evaluation_round_trips() {
    let response = reached(
        client()
            .evaluate(STATEMENT, Classification::questions())
            .await,
    );

    assert!(response.model.starts_with("laya-"), "{}", response.model);
    assert!(
        response
            .request_id
            .as_deref()
            .is_some_and(|id| id.starts_with("laya-")),
        "the server should send x-typesafe-request-id"
    );
    assert!(response.usage.input_tokens.is_some_and(|tokens| tokens > 0));
    assert_eq!(response.unknown().count(), 0, "{:?}", response.answers);

    let classification: Classification = response.extract().expect("the answers read back typed");
    println!("{} answered:\n{classification:#?}", response.model);

    assert!((0.0..=1.0).contains(&classification.is_financial_statements.noul));
    assert_distribution("kind", classification.kind.probabilities.values());
    assert_distribution(
        "completeness",
        classification.completeness.probabilities.values(),
    );
    assert_eq!(
        classification.completeness.legend.len(),
        3,
        "the legend echoes every level that was sent"
    );
}

#[tokio::test]
async fn an_unknown_model_is_unprocessable() {
    let error = client()
        .evaluate(
            STATEMENT,
            questions! { "q" => Noul::new("Is this a test?") },
        )
        .model("not-a-model")
        .await
        .expect_err("an unknown model name should be refused");

    let api = error.api().unwrap_or_else(|| panic!("{error}"));
    assert_eq!(api.kind, ApiErrorKind::Unprocessable, "{error}");
    assert!(
        api.message.as_deref().is_some_and(|m| m.contains("laya")),
        "the error should name the models that are served: {error}"
    );
}

/// With `--jev-alias`, a client left on the crate's default model needs only
/// its base URL changed. Without it, the same client is refused with the fix.
#[tokio::test]
async fn a_client_left_on_jev_is_answered_only_when_the_alias_is_on() {
    let aliased = reached(client().models().await)
        .iter()
        .any(|model| model.name == JEV_LATEST);
    let default_model = builder().build().expect("the client builds");
    let result = default_model
        .evaluate(
            STATEMENT,
            questions! { "q" => Noul::new("Is this a set of financial statements?") },
        )
        .await;

    if aliased {
        let response = reached(result);
        assert!(
            response.model.starts_with("laya-"),
            "the response should name the Laya checkpoint that answered, not {}",
            response.model
        );
    } else {
        let error = result.expect_err("without the alias, a Jev name should be refused");
        assert_eq!(
            error.api().map(|api| api.kind),
            Some(ApiErrorKind::Unprocessable),
            "{error}"
        );
        assert!(error.to_string().contains("--jev-alias"), "{error}");
    }
}

#[tokio::test]
async fn a_long_state_reports_the_truncation() {
    let client = client();
    let ledger = "Sundry expenses 1,250.00 paid by direct debit on account 4411. ".repeat(400);
    let request = client
        .request(
            ledger,
            questions! { "q" => Noul::new("Is this a set of financial statements?") },
        )
        .expect("the request builds");

    // `truncation` is not part of the TypeSafe response, so read the raw body.
    let raw = reached(client.send_raw(&request).await);
    let truncation = &raw["truncation"];
    println!("{truncation:#}");

    let total = truncation["state_tokens"].as_u64().expect("state_tokens");
    let kept = truncation["kept_tokens"]["q"]
        .as_u64()
        .expect("kept_tokens.q");
    assert!(kept < total, "{truncation:#}");
}
