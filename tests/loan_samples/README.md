# Loan Sample Documents

This directory should contain sample loan application documents for testing the loan processing pipeline.

## Expected Document Types

| Document | Type | Description |
|----------|------|-------------|
| `id_sample.pdf` or `id_sample.jpg` | ID | Driver's license, passport, or state ID |
| `w2_sample.pdf` or `w2_sample.jpg` | W2 | W-2 tax form showing annual wages |
| `paystub_sample.pdf` | PAY_STUB | Paycheck stub with pay period and earnings |
| `bank_statement_sample.pdf` | BANK_STATEMENT | Bank account statement with balance |
| `investment_statement_sample.pdf` | INVESTMENT_STATEMENT | Brokerage or retirement account statement |

## Sample Files from LandingAI Course

If you have access to the LandingAI course materials (L9), you can copy:
- `uploadA.pdf` - Bank statement
- `uploadB.pdf` - Investment statement
- `uploadC.pdf` - Pay stub
- `uploadD.jpeg` - ID
- `uploadE.jpg` - W2 form

## Running Tests

```bash
# Run the pipeline on all samples
python -m src.pipelines.loan_processing.pipeline tests/loan_samples/*.pdf tests/loan_samples/*.jpg

# Or use the Streamlit UI
streamlit run ui/app.py
# Navigate to "Loan Processor" page
```

## Creating Test Documents

For testing, you can use:
1. Sample documents from the LandingAI course
2. Publicly available sample financial documents
3. Generated test documents (ensure they contain realistic data)

**Note**: Do not commit real personal financial documents to the repository.
