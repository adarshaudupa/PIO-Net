#include "stage2_infer.h"
#include "model_data.h"
#include <stdio.h>

static float last_pred_score = 0.0f;

float stage2_get_last_score(void) {
    return last_pred_score;
}

int8_t stage2_classify(const float *in) {
    // 1. Conv1: [1, 39, 16] -> [16, 20, 8] into Arena A
    for (int oc = 0; oc < 16; oc++) {
        const float *w_oc = &conv1_weights[oc * 15];
        float b = conv1_biases[oc];
        for (int oh = 0; oh < 20; oh++) {
            for (int ow = 0; ow < 8; ow++) {
                float acc = b;
                for (int kh = 0; kh < 5; kh++) {
                    int ih = oh * 2 - 2 + kh; // stride 2, pad 2
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw; // stride 2, pad 1
                        if (ih >= 0 && ih < 39 && iw >= 0 && iw < 16) {
                            acc += in[ih * 16 + iw] * w_oc[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_a[oc * 160 + oh * 8 + ow] = (acc > 0.0f) ? acc : 0.0f;
            }
        }
    }

    // 2. DS1 Depthwise: [16, 20, 8] -> [16, 20, 8] into Arena B
    for (int c = 0; c < 16; c++) {
        const float *w_c = &ds1_dw_weights[c * 9];
        float b = ds1_dw_biases[c];
        for (int h = 0; h < 20; h++) {
            for (int w = 0; w < 8; w++) {
                float acc = b;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h - 1 + kh; // stride 1, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = w - 1 + kw;
                        if (ih >= 0 && ih < 20 && iw >= 0 && iw < 8) {
                            acc += activation_arena_a[c * 160 + ih * 8 + iw] * w_c[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_b[c * 160 + h * 8 + w] = (acc > 0.0f) ? acc : 0.0f;
            }
        }
    }

    // 3. DS1 Pointwise: [16, 20, 8] -> [24, 20, 8] into Arena A
    for (int oc = 0; oc < 24; oc++) {
        const float *w_oc = &ds1_pw_weights[oc * 16];
        float b = ds1_pw_biases[oc];
        for (int i = 0; i < 160; i++) {
            float acc = b;
            for (int ic = 0; ic < 16; ic++) {
                acc += activation_arena_b[ic * 160 + i] * w_oc[ic];
            }
            activation_arena_a[oc * 160 + i] = (acc > 0.0f) ? acc : 0.0f;
        }
    }

    // 4. DS2 Depthwise: [24, 20, 8] -> [24, 10, 4] into Arena B
    for (int c = 0; c < 24; c++) {
        const float *w_c = &ds2_dw_weights[c * 9];
        float b = ds2_dw_biases[c];
        for (int oh = 0; oh < 10; oh++) {
            for (int ow = 0; ow < 4; ow++) {
                float acc = b;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh; // stride 2, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (ih >= 0 && ih < 20 && iw >= 0 && iw < 8) {
                            acc += activation_arena_a[c * 160 + ih * 8 + iw] * w_c[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_b[c * 40 + oh * 4 + ow] = (acc > 0.0f) ? acc : 0.0f;
            }
        }
    }

    // 5. DS2 Pointwise: [24, 10, 4] -> [32, 10, 4] into Arena A
    for (int oc = 0; oc < 32; oc++) {
        const float *w_oc = &ds2_pw_weights[oc * 24];
        float b = ds2_pw_biases[oc];
        for (int i = 0; i < 40; i++) {
            float acc = b;
            for (int ic = 0; ic < 24; ic++) {
                acc += activation_arena_b[ic * 40 + i] * w_oc[ic];
            }
            activation_arena_a[oc * 40 + i] = (acc > 0.0f) ? acc : 0.0f;
        }
    }

    // 6. DS3 Depthwise: [32, 10, 4] -> [32, 10, 4] into Arena B
    for (int c = 0; c < 32; c++) {
        const float *w_c = &ds3_dw_weights[c * 9];
        float b = ds3_dw_biases[c];
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 4; w++) {
                float acc = b;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h - 1 + kh; // stride 1, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = w - 1 + kw;
                        if (ih >= 0 && ih < 10 && iw >= 0 && iw < 4) {
                            acc += activation_arena_a[c * 40 + ih * 4 + iw] * w_c[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_b[c * 40 + h * 4 + w] = (acc > 0.0f) ? acc : 0.0f;
            }
        }
    }

    // 7. DS3 Pointwise: [32, 10, 4] -> [32, 10, 4] into Arena A
    for (int oc = 0; oc < 32; oc++) {
        const float *w_oc = &ds3_pw_weights[oc * 32];
        float b = ds3_pw_biases[oc];
        for (int i = 0; i < 40; i++) {
            float acc = b;
            for (int ic = 0; ic < 32; ic++) {
                acc += activation_arena_b[ic * 40 + i] * w_oc[ic];
            }
            activation_arena_a[oc * 40 + i] = (acc > 0.0f) ? acc : 0.0f;
        }
    }

    // 8. Global Average Pooling: [32, 10, 4] -> pool[32]
    float pooled[32];
    for (int c = 0; c < 32; c++) {
        float sum = 0.0f;
        for (int i = 0; i < 40; i++) {
            sum += activation_arena_a[c * 40 + i];
        }
        pooled[c] = sum * 0.025f; // (1.0f / 40.0f)
    }

    // 9. Fully Connected Classifier: [32] -> 12 class logits
    int8_t best_class = 0;
    float max_logit = -1e9f;
    for (int cl = 0; cl < 12; cl++) {
        float acc = fc_biases[cl];
        const float *w_cl = &fc_weights[cl * 32];
        for (int i = 0; i < 32; i++) {
            acc += pooled[i] * w_cl[i];
        }
        if (acc > max_logit) {
            max_logit = acc;
            best_class = (int8_t)cl;
        }
    }

    last_pred_score = max_logit;
    return best_class;
}