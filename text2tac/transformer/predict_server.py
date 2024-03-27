import contextlib
import sys
import socket
import socketserver
import argparse
import pytact.graph_visualize as gv
from pytact.data_reader import (capnp_message_generator, ProofState,
                                TacticPredictionGraph, TacticPredictionsGraph,
                                TacticPredictionText, TacticPredictionsText,
                                GlobalContextMessage, CheckAlignmentMessage, CheckAlignmentResponse)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast
from transformers import GPT2Config
import argparse

import torch

def log_normalize(x):

    temp = torch.logsumexp(x, dim=0)

    return x - temp


from transformers import AutoTokenizer, T5ForConditionalGeneration

def load_eval_setup(model_location, device):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_location)
    model = model.to(device)
    return tokenizer, model

# def load_eval_setup(toksave, model_location, truncate_side, device, emb_size, n_layers):
#     """"
#     Load a tokenizer and a model from the specified locations.


#     """

#     tokenizer = GPT2TokenizerFast.from_pretrained(toksave, max_len=1024)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     tokenizer.add_special_tokens({'eos_token': '<END>'})
#     tokenizer.add_special_tokens({'bos_token': '<END>'})

#     config = GPT2Config(vocab_size=3003,
#                         n_positions=1024,
#                         n_embd=emb_size,
#                         n_head=12,
#                         n_layer=n_layers,
#                         eos_token_id=tokenizer(["<END>"]).input_ids[0][0],
#                         bos_token_id=tokenizer(["<END>"]).input_ids[0][0],
#                         pad_token_id=tokenizer(["[PAD]"]).input_ids[0][0]
#                         )

#     model = GPT2LMHeadModel(config=config)

#     model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))
#     model.eval()

#     model = model.to(device)

#     if truncate_side == "left":
#         tokenizer.truncation_side = 'left'

#     return tokenizer, model


def generate(input_proof_state, tokenizer, model, device, sampling_on, beam_w, topk, temperature, logspace, log_base):
    """
    Beam search with width beam_w, returns best beam_w sequences.

    Input:

    - string version of proofstate
    - tokenizer instance
    - gpt2 model instance

    Output:

    - [String]
    """

    sample = input_proof_state


    input_ids = tokenizer(sample, truncation=True, max_length=970, return_tensors="pt", padding=False).input_ids.to(
        device)

    # if input_ids.shape[1] > 1024:
    #    input_ids = input_ids[:, :1023]
    # raise ValueError("Input is too long")
    if not sampling_on:

        beam_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            num_beams=beam_w,
            early_stopping=True,
            num_return_sequences=beam_w,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=0,
        )
    else:
        beam_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            num_beams=beam_w,
            early_stopping=True,
            num_return_sequences=beam_w,
            do_sample=True,
            output_scores=True,
            top_k=topk,
            return_dict_in_generate=True,
            length_penalty=0,
            temperature=temperature,
        )

    return_list = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=False) for o in beam_output["sequences"]]

    print(f"XXXX {return_list}")

    # for e, i in enumerate(beam_output['sequences']):
    #     # print(input_ids.shape[1])
    #     # print(tokenizer.decode(i[input_ids.shape[1]:], skip_special_tokens=True))
    #     # print(tokenizer.decode(i, skip_special_tokens=True))
    #     model_suggestion = tokenizer.decode(i[input_ids.shape[1]:], skip_special_tokens=True,
    #                                         clean_up_tokenization_spaces=False)
    #     model_suggestion = model_suggestion.rstrip()
    #     model_suggestion = model_suggestion.lstrip()
    #     return_list.append(model_suggestion)
        
    if not logspace:
        seq_probs = torch.exp(beam_output['sequences_scores'])

        normalized_sequence_probs = seq_probs / torch.sum(seq_probs)
    else:
        if log_base == -1:
            # ln base
            normalized_sequence_probs = log_normalize(beam_output['sequences_scores'])
        else:
            # change of base: log_2 (x) = log_e(x) / log_e(2)
            normalized_sequence_probs = log_normalize(beam_output['sequences_scores']) / torch.log(
                torch.tensor(log_base))


    return return_list, normalized_sequence_probs.tolist()




def text_prediction_loop(context : GlobalContextMessage, generate_args):
    tactics = [ 'idtac "is it working?"', 'idtac "yes it is working!"', 'auto' ]
    prediction_requests = context.prediction_requests

    tokenizer, model, device, sampling_on, beam_w, topk, temperature, logspace, log_base = generate_args

    for msg in prediction_requests:
        if isinstance(msg, ProofState):
            proof_state = msg
            print(proof_state.text)

            tactics, probs = generate(proof_state.text.lstrip(), tokenizer, model, device, sampling_on, beam_w, topk, temperature, logspace, log_base)
            preds = [TacticPredictionText(t, p) for (t, p) in zip(tactics, probs)]

            prediction_requests.send(TacticPredictionsText(preds))
        elif isinstance(msg, CheckAlignmentMessage):
            alignment = CheckAlignmentResponse([], [])
            prediction_requests.send(alignment)
        elif isinstance(msg, GlobalContextMessage):
            text_prediction_loop(msg, generate_args)
        else:
            raise Exception("Capnp protocol error")


def run_session(args, capnp_socket, record_file, generate_args):
    messages_generator = capnp_message_generator(capnp_socket, record_file)

    print('Python server running in text mode')
    text_prediction_loop(messages_generator, generate_args)


def main():
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(
        description = "Text2tac python server capable of communicating with Coq through Tactician's 'synth' tactic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--tcp',
                        type = int,
                        default = 0,
                        help='Run in tcp mode instead of stdin mode on the specified port.')
    parser.add_argument('--record',
                        dest="record_file",
                        type = str,
                        default = None,
                        help='Record all exchanged messages to the specified file, so that they can later be ' +
                        'replayed through "pytact-fake-coq"')

    parser.add_argument('--model', type=str, help="Location of the transformer .bin file")

    # parser.add_argument('--tokenizer', type=str, help="Location of the tokenizer files (folder)")

    parser.add_argument('--truncate_side', type=str, help="left or right truncation of proof states that are too long",
                        default="right")

    parser.add_argument('--dev', type=str, help="cuda | cpu", default="cuda")


    parser.add_argument('--pt_threads', type=int, help="maximum threads for PyTorch", default=1)

    parser.add_argument('--beam_width', type=int, default=10, help="How many alternative answers to keep in beam")

    # parser.add_argument('--emb_size', type=int, help="embeddin size", default=768)
    # parser.add_argument('--layers', type=int, help="transformer layers", default=12)

    parser.add_argument('--temp', type=float, help="temperature for softmax", default=1.0)
    parser.add_argument('--sample', type=int, help="use top-k sampling | off by default", default=0)
    parser.add_argument('--topk', type=int,
                        help="number of tokens to take into account in top_k sampling - needs to be set if --sample is on.",
                        default=-1)



    args = parser.parse_args()

    model_location = args.model
    # tokenizer_location = args.tokenizer
    truncate_side = args.truncate_side
    n_threads = args.pt_threads
    # n_layers = args.layers
    # emb_size = args.emb_size

    beam_w = args.beam_width

    temperature = args.temp

    sampling_on = bool(args.sample)
    topk = args.topk

    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)

    logspace = True
    log_base = -1 # default base e

    if topk == -1 and sampling_on:
        raise ValueError("you turned on top k sampling, supply a number to --topk; for example 20")

    # tokenizer, model = load_eval_setup(tokenizer_location, model_location, truncate_side, args.dev, emb_size, n_layers)
    tokenizer, model = load_eval_setup(model_location, args.dev)


    generate_args = (tokenizer, model, args.dev, sampling_on, beam_w, topk, temperature, logspace, log_base)

    if args.record_file is not None:
        record_context = open(args.record_file, 'wb')
    else:
        record_context = contextlib.nullcontext()
    with record_context as record_file:
        if args.tcp != 0:
            addr = ('0.0.0.0', args.tcp)
            if socket.has_dualstack_ipv6():
                try:
                    server_sock = socket.create_server(
                        addr,
                        family=socket.AF_INET6, dualstack_ipv6=True
                    )
                except OSError:
                    server_sock = socket.create_server(addr)
            else:
                server_sock = socket.create_server(addr)
            try:
                server_sock.listen(1)
                while True:
                    capnp_socket, remote_addr = server_sock.accept()
                    run_session(args, capnp_socket, record_file, generate_args)
                    # predict_server.start_prediction_loop(capnp_socket, record_file)
            finally:
                server_sock.close()
        else:
            capnp_socket = socket.socket(fileno=sys.stdin.fileno())
            run_session(args, capnp_socket, record_file, generate_args)

if __name__ == '__main__':
    main()
