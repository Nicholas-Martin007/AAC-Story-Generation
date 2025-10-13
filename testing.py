from evaluate import load

bertscore = load('bertscore')
result = bertscore.compute(
    predictions=['saya makan nasi'],
    references=['aku lagi makan nasi'],
    lang='id',
    model_type='bert-base-multilingual-cased',
)

print(result)
