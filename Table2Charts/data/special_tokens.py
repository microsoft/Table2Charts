from .config import DataConfig
from .token import Token, TokenType, AggFunc, AnaType, GroupingOp
from .util import load_json


class SpecialTokens:
    ANA_TOKEN = Token(TokenType.ANA)  # Be careful, this is [PivotTable]
    PAD_TOKEN = Token(TokenType.PAD)
    SEP_TOKEN = Token(TokenType.SEP)
    GRP_OP_TOKENS = [Token(TokenType.GRP, grp_op=GroupingOp.Cluster),
                     Token(TokenType.GRP, grp_op=GroupingOp.Stack)]

    def __init__(self, config: DataConfig):
        self.AGG_FUNC_TOKENS = []
        self.AGG_FUNC_DICT = dict()
        func_embeds = load_json("data/agg_func.EMB.json", encoding=config.encoding)

        index = 0
        for func in AggFunc:
            embed = func_embeds[index][config.embed_layer][config.embed_reduce_type] \
                if config.use_semantic_embeds else None
            self.AGG_FUNC_TOKENS.append(Token(TokenType.FUNC, semantic_embedding=embed, agg_func=func))
            self.AGG_FUNC_DICT[func] = self.AGG_FUNC_TOKENS[-1]
            index += 1

    def get_func_token(self, agg_func: AggFunc):
        return self.AGG_FUNC_DICT[agg_func]

    @classmethod
    def get_grp_token(cls, grp_op: GroupingOp):
        if grp_op is GroupingOp.Cluster:
            return cls.GRP_OP_TOKENS[0]
        else:  # Stack
            return cls.GRP_OP_TOKENS[1]

    @staticmethod
    def get_ana_token(ana_type: AnaType = AnaType.PivotTable):
        return Token(TokenType.ANA, ana_type=ana_type)
