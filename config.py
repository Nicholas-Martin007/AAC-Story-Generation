import torch

N_PERSON = [
    'no_person',
    # 'one_person',
    # 'two_person',
    # 'three_person',
    # 'four_person',
    # 'five_person',
]

AAC_CARD_PATH = 'dataset/Oktober_24_2025/aac_card_dataset.json'
AAC_STORY_PATH = 'dataset/Oktober_24_2025/aac_story_dataset.json'

NER_MODEL_PATH = 'cahya/NusaBert-ner-v1.3'
DEVICE = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

SDG_MODEL_PATH = 'meta-llama/Llama-3.2-1B-Instruct'


MODEL_PATH = {
    'llama3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',  # 1.24B parameters
    'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',  # 3.21B parameters
    'mistral7b': 'mistralai/Mistral-7B-Instruct-v0.3',  # 7.25B parameters
    'flan-large': 'google/flan-t5-large',  # 783M parameters
    'flan-xl': 'google/flan-t5-xl',  # 2.85B parameters
}

FINE_TUNE_OUTPUT_DIR = './training_output'

SEED = 42

EVALUATION_OUTPUT_DIR = './evaluation_results'

FINE_TUNED_MODELS = {
    # 'llama': {
    #     'path': MODEL_PATH['llama3.2-3b'],
    #     'model_type': 'causal',
    #     'qlora_model': [
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr0_0001135780665671859_wd0_06_r16_a32_ep1_bs2',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr0_00025461989761985766_wd0_01_r48_a24_ep2_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr1_4421754150410843e-05_wd0_02_r48_a96_ep1_bs1',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr1_4731842491352058e-05_wd0_01_r64_a32_ep3_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr3_093713826482805e-05_wd0_09_r64_a128_ep2_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr3_450484311915849e-05_wd0_0_r48_a96_ep2_bs1',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr4_8124593091828134e-05_wd0_07_r48_a96_ep3_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr0_0002718408170704472_wd0_04_r32_a64_ep3_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr0_00013526389629795978_wd0_01_r64_a32_ep4_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr0_00035639553982565013_wd0_08_r16_a8_ep4_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr1_4731842491352058e-05_wd0_01_r64_a32_ep3_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr3_68892840948597e-05_wd0_06_r16_a32_ep4_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr4_8124593091828134e-05_wd0_07_r48_a96_ep3_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr9_218303817885077e-05_wd0_05_r48_a96_ep4_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr0_00034608371233975127_wd0_0_r32_a16_ep2_bs2',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr2_5343481047399046e-05_wd0_06_r48_a96_ep1_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr3_533756450157008e-05_wd0_0_r48_a96_ep1_bs2',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr5_5147579616758885e-05_wd0_03_r64_a32_ep1_bs2',
    #         '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr8_864904332323238e-05_wd0_04_r32_a16_ep2_bs8',
    #     ],
    # },
    # 'mistral': {
    #     'path': MODEL_PATH['mistral7b'],
    #     'model_type': 'causal',
    #     'qlora_model': [
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_0003159894127911858_wd0_03_r48_a96_ep2_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_00015079044135156433_wd0_1_r64_a128_ep1_bs2',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_00015249001841056552_wd0_05_r64_a128_ep1_bs1',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr1_1074139343117776e-05_wd0_05_r64_a32_ep2_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr2_303555912500907e-05_wd0_0_r48_a24_ep1_bs1',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_0001396804189969797_wd0_03_r16_a32_ep3_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_0004615618598756448_wd0_09_r48_a96_ep4_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr1_716057648401377e-05_wd0_02_r32_a16_ep4_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr6_389384276447428e-05_wd0_06_r48_a24_ep3_bs2',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr7_94899495911881e-05_wd0_05_r32_a16_ep3_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_0002560675341706887_wd0_01_r16_a8_ep2_bs4',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_00012932051696890756_wd0_02_r48_a96_ep1_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_00015034525856502643_wd0_06_r64_a128_ep1_bs8',
    #         # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr0_00031777491797078413_wd0_04_r16_a8_ep2_bs4',
    #         '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/mistral7b_lr5_123042645826949e-05_wd0_03_r32_a16_ep2_bs4',
    #     ],
    # },
    'flan': {
        'path': MODEL_PATH['flan-large'],
        'model_type': 'seq2seq',
        'qlora_model': [
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr0_0001482178201997769_wd0_09_r32_a16_ep1_bs4',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr0_00042965736678486025_wd0_03_r16_a8_ep2_bs8',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr1_1161741857853943e-05_wd0_04_r16_a8_ep1_bs2',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr1_5200303235995328e-05_wd0_01_r16_a32_ep1_bs4',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr1_6937795180734888e-05_wd0_01_r48_a24_ep1_bs8',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr0_0002970514720912085_wd0_03_r32_a16_ep3_bs2',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr0_0003411262333753138_wd0_08_r64_a128_ep3_bs4',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr0_0004133269026741804_wd0_02_r48_a24_ep3_bs2',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr3_067200164436362e-05_wd0_07_r64_a32_ep4_bs2',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr3_674854547639234e-05_wd0_1_r32_a64_ep4_bs8',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr0_00012798675705999834_wd0_05_r16_a8_ep2_bs2',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr0_00021843616018293275_wd0_04_r32_a64_ep2_bs2',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr1_222853052162347e-05_wd0_02_r16_a32_ep1_bs4',
            # '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr4_317348906262374e-05_wd0_08_r64_a128_ep2_bs2',
            '/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/flan-large_lr7_789020928835203e-05_wd0_07_r16_a32_ep2_bs4',
        ],
    },
}
