from typing import List, Tuple

import torch

from decoding.earley_partial_parse import GrammarTokenizerInfo, UTF8EarleyPartialParse
from decoding.uint8_earley_partial_parse import UInt8EarleyPartialParse
from domains.sql.cosql.grammar import grammar_for_schema, load_base_grammar, preprocessed_grammar_for_schema
from domains.sql.cosql.schema import DbSchema, Table, Column, ColumnType
from scfg.generate import parse_and_render
from scfg.read_grammar import PreprocessedGrammar
from scfg.scfg import SCFG
from tokenization import GPT2ClampTokenizer

INSTRUCTOR_SCHEMA = DbSchema(
    "instructor",
    [
        Table(
            "instructor",
            [
                Column("salary", ColumnType.Number),
                Column("dept_name", ColumnType.Text),
                Column("name", ColumnType.Text),
            ],
        ),
    ],
)



# This grammar can generate:
#   abcA
#   abcB
#   abcC
#   abcDE
SIMPLE_SQL_GRAMMAR = """
start -> a
start -> b
start -> c
a -> "a" "bcA"
a -> "ab" "cB"
b -> "abc" "C"
b -> c "DE"
c -> "a" "bc"
"""
SIMPLE_GRAMMAR = """
start -> select_star_from, select_star_from
select_star_from -> "SELECT * FROM" space_table , "SELECT * FROM" space_table
space_table -> " " table , " " table
b -> "abc" "C" , "abcC"
b -> c "DE" , "abcDE"
c -> "a" "bc" , "abc"
"""

if __name__ == '__main__':
    # some programs from CoSQL
    sql_gram = grammar_for_schema(INSTRUCTOR_SCHEMA, load_base_grammar())
    utterance: str = "SElECT avg ( salary )  FROM instructor"
    tokenizer: GPT2ClampTokenizer = GPT2ClampTokenizer.from_pretrained("gpt2")
    grammar: PreprocessedGrammar = PreprocessedGrammar.from_line_iter(SIMPLE_GRAMMAR.splitlines())
    grammar_tok_info: GrammarTokenizerInfo = GrammarTokenizerInfo.create(tokenizer, grammar, True)
    partial_parse = UTF8EarleyPartialParse.initial(grammar_tok_info, "a")
    all = torch.tensor(range(tokenizer.vocab_size))
    allowed_next = partial_parse.allowed_next(all, enable_fallback_from_grammar=True)
    allowed_next