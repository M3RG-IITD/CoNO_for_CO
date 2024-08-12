from models import complex_FFNO_2D, FNO_2D

def get_model(args):
    model_dict = {
        'FNO_2D': FNO_2D,
        'complex_FFNO_2D': complex_FFNO_2D
    }
    
    return model_dict[args.model].Model(args).cuda()