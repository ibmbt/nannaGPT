#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include "model_weights.h"

using namespace std;

// --- HYPERPARAMETERS ---
const int vocab_size = 65;
const int embedding_dim = 384;  // Increased to 16
const int num_heads = 6;       // 4 parallel heads
const int head_size = embedding_dim / num_heads; // 16 / 4 = 4
const int block_size = 256;
const int hidden_dim = 4 * embedding_dim;
const int K = 4;
const float temperature = 0.7f; // 1.0 is creative/messy, 0.1 is strict/repetitive

vector<int> getUserPrompt(const vector<char>& vocab) {
    string user_input;
    cout << "starting prompt: ";

    // We use getline instead of cin >> so it captures spaces properly
    getline(cin, user_input);

    vector<int> tokens;

    // Loop through every character the user typed
    for (char c : user_input) {
        bool found = false;

        // Find the matching ID in our dictionary
        for (int i = 0; i < vocab.size(); i++) {
            if (vocab[i] == c) {
                tokens.push_back(i);
                found = true;
                break;
            }
        }

        // Safety net for characters outside the 65-char vocabulary
        if (!found) {
            cout << "[Warning] The character '" << c << "' is not in the model's vocabulary. Skipping.\n";
        }
    }

    // If the user just hit enter without typing anything, give the engine a newline token to start with
    if (tokens.empty()) {
        tokens.push_back(0); // Index 0 is '\n' in your dictionary
    }

    return tokens;
}

vector<char> itos = {
    '\n', ' ', '!', '$', '&', '\'', ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
     'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };

// --- CORE DATA STRUCTURES ---
struct Matrix {
    int rows, cols;
    vector<vector<float>> matrix;

    Matrix(int r, int c) : rows(r), cols(c) {
        matrix.resize(r, vector<float>(c, 0.0f));
    }

    void loadFromArray(const float* arr) {
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = arr[idx++];
            }
        }
    }

};

// Groups all weights belonging to a single Transformer Block
struct LayerWeights {
    Matrix W_q, W_k, W_v, W_o; // Added W_o here
    Matrix W_1, b_1, W_2, b_2;

    LayerWeights() :
        W_q(embedding_dim, embedding_dim), W_k(embedding_dim, embedding_dim),
        W_v(embedding_dim, embedding_dim), W_o(embedding_dim, embedding_dim), // Init W_o
        W_1(embedding_dim, hidden_dim), b_1(1, hidden_dim),
        W_2(hidden_dim, embedding_dim), b_2(1, embedding_dim) {
    }
};


// Groups the global input and output weights
struct TransformerWeights {
    Matrix W_token;
    Matrix W_pos;
    Matrix W_out;

    TransformerWeights() :
        W_token(vocab_size, embedding_dim),
        W_pos(block_size, embedding_dim),
        W_out(embedding_dim, vocab_size) {
    }
};

// --- MATH UTILITIES ---
Matrix add(Matrix a, Matrix b) {
    Matrix c(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            c.matrix[i][j] = a.matrix[i][j] + b.matrix[i][j];
        }
    }
    return c;
}

Matrix addBias(Matrix a, Matrix bias) {
    Matrix c(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            c.matrix[i][j] = a.matrix[i][j] + bias.matrix[0][j];
        }
    }
    return c;
}

Matrix multiply(Matrix a, Matrix b) {
    Matrix c(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++) {
                sum += (a.matrix[i][k] * b.matrix[k][j]);
            }
            c.matrix[i][j] = sum;
        }
    }
    return c;
}

Matrix transpose(Matrix a) {
    Matrix c(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            c.matrix[j][i] = a.matrix[i][j];
        }
    }
    return c;
}

Matrix division(Matrix a, float x) {
    Matrix c(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            c.matrix[i][j] = a.matrix[i][j] / x;
        }
    }
    return c;
}

Matrix reLu(Matrix y) {
    for (int i = 0; i < y.rows; i++) {
        for (int j = 0; j < y.cols; j++) {
            if (y.matrix[i][j] < 0) {
                y.matrix[i][j] = 0;
            }
        }
    }
    return y;
}

void softmax(Matrix& x) {
    for (int i = 0; i < x.rows; i++) {
        float expSum = 0.0f;
        for (int j = 0; j < x.cols; j++) {
            expSum += expf(x.matrix[i][j]);
        }
        for (int j = 0; j < x.cols; j++) {
            x.matrix[i][j] = expf(x.matrix[i][j]) / expSum;
        }
    }
}

Matrix layerNorm(Matrix x, const float* ln_weight) {
    for (int i = 0; i < x.rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < x.cols; j++) { sum += x.matrix[i][j]; }
        float mean = sum / x.cols;

        float varSum = 0.0f;
        for (int j = 0; j < x.cols; j++) { varSum += (x.matrix[i][j] - mean) * (x.matrix[i][j] - mean); }
        float var = varSum / x.cols;

        for (int j = 0; j < x.cols; j++) {
            // Apply normalization, the 1e-5f safety net, AND the PyTorch weight
            x.matrix[i][j] = ((x.matrix[i][j] - mean) / sqrt(var + 1e-5f)) * ln_weight[j];
        }
    }
    return x;
}

// --- NEURAL NETWORK BLOCKS ---

Matrix embedTokens(vector<int> tokens, TransformerWeights& weights) {
    int T = tokens.size();
    Matrix x(T, embedding_dim);
    for (int i = 0; i < T; i++) {
        int token_id = tokens[i];
        for (int j = 0; j < embedding_dim; j++) {
            x.matrix[i][j] = weights.W_token.matrix[token_id][j];
        }
    }
    return x;
}

Matrix addPositionalEmbedding(Matrix x, TransformerWeights& weights) {
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            x.matrix[i][j] += weights.W_pos.matrix[i][j];
        }
    }
    return x;
}

Matrix selfAttention(Matrix x, LayerWeights& layer) {
    Matrix Q = multiply(x, layer.W_q);
    Matrix K = multiply(x, layer.W_k);
    Matrix V = multiply(x, layer.W_v);

    Matrix K_T = transpose(K);
    Matrix scores = multiply(Q, K_T);

    scores = division(scores, sqrt(embedding_dim));

    for (int i = 0; i < scores.rows; i++) {
        for (int j = 0; j < scores.cols; j++) {
            if (j > i) {
                scores.matrix[i][j] = -1e9;
            }
        }
    }

    softmax(scores);

    Matrix attn_out = multiply(scores, V);
    return add(attn_out, x);
}

Matrix multiHeadAttention(Matrix x, LayerWeights& layer) {
    // 1. Generate massive Q, K, V for all heads at once
    Matrix Q = multiply(x, layer.W_q);
    Matrix K = multiply(x, layer.W_k);
    Matrix V = multiply(x, layer.W_v);

    Matrix concatenated_out(x.rows, x.cols); // Holds the glued-together results

    // 2. Loop through each head
    for (int h = 0; h < num_heads; h++) {
        int start_col = h * head_size;

        // Create empty matrices for this specific head
        Matrix Q_h(x.rows, head_size);
        Matrix K_h(x.rows, head_size);
        Matrix V_h(x.rows, head_size);

        // Slice the massive matrices to get just this head's channels
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < head_size; j++) {
                Q_h.matrix[i][j] = Q.matrix[i][start_col + j];
                K_h.matrix[i][j] = K.matrix[i][start_col + j];
                V_h.matrix[i][j] = V.matrix[i][start_col + j];
            }
        }

        // Run the standard attention math on this isolated slice
        Matrix K_h_T = transpose(K_h);
        Matrix scores = multiply(Q_h, K_h_T);

        // Divide by sqrt(head_size)
        scores = division(scores, sqrt(head_size));

        // Masking
        for (int i = 0; i < scores.rows; i++) {
            for (int j = 0; j < scores.cols; j++) {
                if (j > i) scores.matrix[i][j] = -1e9;
            }
        }

        softmax(scores);

        // Context for this head
        Matrix head_out = multiply(scores, V_h);

        // Paste the result back into the master output matrix
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < head_size; j++) {
                concatenated_out.matrix[i][start_col + j] = head_out.matrix[i][j];
            }
        }
    }

    // 3. Final Projection (Mix the heads together)
    Matrix final_attn = multiply(concatenated_out, layer.W_o);

    // Return the raw output (Residual add happens in main)
    return final_attn;
}

Matrix FFN(Matrix x, LayerWeights& layer) {
    Matrix z = multiply(x, layer.W_1);
    z = addBias(z, layer.b_1);
    z = reLu(z);

    Matrix ffn_out = multiply(z, layer.W_2);
    ffn_out = addBias(ffn_out, layer.b_2);

    // Return the raw output (Residual add happens in main)
    return ffn_out;
}

Matrix finalLinearLayer(Matrix x, TransformerWeights& weights) {
    Matrix logits = multiply(x, weights.W_out);

    // --- THE TEMPERATURE DIAL ---


    for (int i = 0; i < logits.rows; i++) {
        for (int j = 0; j < logits.cols; j++) {
            logits.matrix[i][j] = logits.matrix[i][j] / temperature;
        }
    }
    // -----------------------------

    softmax(logits);
    return logits;
}

struct TokenProb {
    int id;
    float prob;
};


int main() {
    // 1. INIT WEIGHTS
    TransformerWeights model_weights;
    LayerWeights layers[6]; // We now have 6 layers!

    cout << "Loading PyTorch Weights...\n";

    // Embeddings
    model_weights.W_token.loadFromArray((const float*)transformer_wte_weight);
    model_weights.W_pos.loadFromArray((const float*)transformer_wpe_weight);

    // Grouping the PyTorch arrays into pointers so we can loop through them
    const float* c_attn[6] = { (const float*)transformer_h_0_attn_c_attn_weight,
        (const float*)transformer_h_1_attn_c_attn_weight,
        (const float*)transformer_h_2_attn_c_attn_weight,
        (const float*)transformer_h_3_attn_c_attn_weight,
        (const float*)transformer_h_4_attn_c_attn_weight,
        (const float*)transformer_h_5_attn_c_attn_weight };
    const float* c_proj[6] = { (const float*)transformer_h_0_attn_c_proj_weight,
        (const float*)transformer_h_1_attn_c_proj_weight,
        (const float*)transformer_h_2_attn_c_proj_weight,
        (const float*)transformer_h_3_attn_c_proj_weight,
        (const float*)transformer_h_4_attn_c_proj_weight,
        (const float*)transformer_h_5_attn_c_proj_weight };
    const float* mlp_fc[6] = { (const float*)transformer_h_0_mlp_c_fc_weight,
        (const float*)transformer_h_1_mlp_c_fc_weight,
        (const float*)transformer_h_2_mlp_c_fc_weight,
        (const float*)transformer_h_3_mlp_c_fc_weight,
        (const float*)transformer_h_4_mlp_c_fc_weight,
        (const float*)transformer_h_5_mlp_c_fc_weight };
    const float* mlp_proj[6] = { (const float*)transformer_h_0_mlp_c_proj_weight,
        (const float*)transformer_h_1_mlp_c_proj_weight,
        (const float*)transformer_h_2_mlp_c_proj_weight,
        (const float*)transformer_h_3_mlp_c_proj_weight,
        (const float*)transformer_h_4_mlp_c_proj_weight,
        (const float*)transformer_h_5_mlp_c_proj_weight };

    const float* ln_1_weights[6] = { (const float*)transformer_h_0_ln_1_weight,
        (const float*)transformer_h_1_ln_1_weight,
        (const float*)transformer_h_2_ln_1_weight,
        (const float*)transformer_h_3_ln_1_weight,
        (const float*)transformer_h_4_ln_1_weight,
        (const float*)transformer_h_5_ln_1_weight };
    const float* ln_2_weights[6] = { (const float*)transformer_h_0_ln_2_weight,
        (const float*)transformer_h_1_ln_2_weight,
        (const float*)transformer_h_2_ln_2_weight,
        (const float*)transformer_h_3_ln_2_weight,
        (const float*)transformer_h_4_ln_2_weight,
        (const float*)transformer_h_5_ln_2_weight };

    // Load, slice, and transpose all 6 layers
    for (int l = 0; l < 6; l++) {
        // Unpack Fused Attention
        for (int i = 0; i < 384; i++) {
            for (int j = 0; j < 384; j++) {
                layers[l].W_q.matrix[i][j] = c_attn[l][j * 384 + i];
                layers[l].W_k.matrix[i][j] = c_attn[l][(j + 384) * 384 + i];
                layers[l].W_v.matrix[i][j] = c_attn[l][(j + 768) * 384 + i];
                layers[l].W_o.matrix[i][j] = c_proj[l][j * 384 + i];
            }
        }
        // Unpack FFN
        for (int i = 0; i < 384; i++) {
            for (int j = 0; j < 1536; j++) {
                layers[l].W_1.matrix[i][j] = mlp_fc[l][j * 384 + i];
            }
        }
        for (int i = 0; i < 1536; i++) {
            for (int j = 0; j < 384; j++) {
                layers[l].W_2.matrix[i][j] = mlp_proj[l][j * 1536 + i];
            }
        }
    }

    // Final Output Layer
    for (int i = 0; i < 384; i++) {
        for (int j = 0; j < 65; j++) {
            model_weights.W_out.matrix[i][j] = transformer_wte_weight[j][i];
        }
    }

    // 2. THE STARTING PROMPT
    // vector<int> input_tokens = { 18, 53, 56 }; // "For"

    // cout << "Starting Prompt: ";
    // for (int t : input_tokens) cout << itos[t];
    // cout << "\nGenerating:\n";

    vector<int> input_tokens = getUserPrompt(itos);

    cout << "\nGenerating:\n";


    // 3. THE AUTOREGRESSIVE LOOP
    int tokens_to_generate = 10000;

    for (int step = 0; step < tokens_to_generate; step++) {
        int T = input_tokens.size();

        if (T >= block_size) {
            cout << "\n\n[Hit Block Size Limit. Generation Stopped.]\n";
            break;
        }

        // --- THE PIPELINE ---
        Matrix X = embedTokens(input_tokens, model_weights);
        X = addPositionalEmbedding(X, model_weights);

        // Run the data through all 6 layers sequentially
        for (int l = 0; l < 6; l++) {
            Matrix norm_x1 = layerNorm(X, ln_1_weights[l]);
            Matrix attn_out = multiHeadAttention(norm_x1, layers[l]);
            X = add(X, attn_out);

            Matrix norm_x2 = layerNorm(X, ln_2_weights[l]);
            Matrix ffn_out = FFN(norm_x2, layers[l]);
            X = add(X, ffn_out);
        }

        Matrix final_norm = layerNorm(X, transformer_ln_f_weight);
        Matrix final_probs = finalLinearLayer(final_norm, model_weights);

        // --- PREDICTION (Top-K Sampling) ---

        // 1. Gather all probabilities for the last token into a list
        vector<TokenProb> probs;
        for (int c = 0; c < vocab_size; c++) {
            probs.push_back({ c, final_probs.matrix[T - 1][c] });
        }

        // 2. Sort the list from highest probability to lowest
        sort(probs.begin(), probs.end(), [](const TokenProb& a, const TokenProb& b) {
            return a.prob > b.prob;
            });

        // 3. Find the total probability mass of just the Top K choices
        float sum_prob = 0.0f;
        for (int i = 0; i < K; i++) {
            sum_prob += probs[i].prob;
        }

        // 4. Spin the Roulette Wheel
        // Generate a random float between 0.0 and sum_prob
        float r = ((float)rand() / RAND_MAX) * sum_prob;
        float cumulative = 0.0f;
        int best_token_id = probs[0].id; // Fallback to the best choice just in case

        // Walk through the Top K. The bigger the probability, the larger its "slice" of the wheel.
        for (int i = 0; i < K; i++) {
            cumulative += probs[i].prob;
            if (r <= cumulative) {
                best_token_id = probs[i].id;
                break; // We found our winner!
            }
        }

        // --- APPEND AND REPEAT ---
        input_tokens.push_back(best_token_id);
        cout << itos[best_token_id] << flush;
    }

    cout << "\n\nFinal Output String: ";
    for (int t : input_tokens) cout << itos[t];
    cout << "\n";

    return 0;
}
