

def get_nets(input_shape, num_classes, num_domains, hparams, args, net_whole=False):
    if args.nets_base in ['bed_nets','domainbed']:
        from networks import bed_nets as nets
        if net_whole:
            return nets
        else:
            featurizer = nets.Featurizer(input_shape, hparams)
            n_outputs = featurizer.n_outputs * 2 if args.algorithm in ['MTL'] else featurizer.n_outputs
            classifier = nets.Classifier(n_outputs,  num_classes)
            return featurizer, classifier

    elif args.nets_base in ['diag_nets','LeNet', 'Transformer']:
        from networks import diag_nets as nets
        if net_whole:
            return nets
        else:
            featurizer = nets.Featurizer(input_shape, hparams, args)
            n_outputs = featurizer.n_outputs * 2 if args.algorithm in ['MTL'] else featurizer.n_outputs
            classifier = nets.Classifier(n_outputs, num_classes)
            return featurizer, classifier

    else:
        raise NotImplementedError # future

