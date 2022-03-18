from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from multilingual_srl.data.conll_data_module import ConllDataModule
from multilingual_srl.models.probers.propbank_structure_prober import PropbankStructureProber
from multilingual_srl.models.probers.verbatlas_structure_prober import VerbatlasStructureProber

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add seed arg.
    parser.add_argument('--seed', type=int, default=313)

    # Add data-specific args.
    parser = ConllDataModule.add_data_specific_args(parser)

    # Add model-specific args.
    parser = PropbankStructureProber.add_model_specific_args(parser)

    # Add all the available trainer options to argparse.
    parser = Trainer.add_argparse_args(parser)

    # Set default arguments.
    parser.set_defaults(
        max_epochs=30,
        gpus=1,
        precision=16,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Store the arguments in args.
    args = parser.parse_args()

    seed_everything(args.seed)

    data_module = ConllDataModule(
        inventory=args.inventory,
        span_based=args.span_based,
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        language_model_type=args.language_model_type,
        language_model_name=args.language_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    data_module.prepare_data()
    data_module.setup('fit')

    if args.inventory == 'propbank':
        model = PropbankStructureProber(
            num_senses=data_module.num_senses,
            num_roles=data_module.num_roles,
            padding_label_id=data_module.padding_label_id,

            language_model_type=args.language_model_type,
            language_model_name=args.language_model_name,
            language_model_fine_tuning=args.language_model_fine_tuning,
            language_model_random_initialization=args.language_model_random_initialization,
            return_static_embeddings=args.return_static_embeddings,
            word_encoding_size=args.word_encoding_size,
            word_encoding_activation=args.word_encoding_activation,
            word_encoding_dropout=args.word_encoding_dropout,

            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            weight_decay=args.weight_decay,
            language_model_learning_rate=args.language_model_learning_rate,
            language_model_min_learning_rate=args.language_model_min_learning_rate,
            language_model_weight_decay=args.language_model_weight_decay
        )
    
    elif args.inventory == 'verbatlas':
        model = VerbatlasStructureProber(
            num_senses=data_module.num_senses,
            num_roles=data_module.num_roles,
            padding_label_id=data_module.padding_label_id,

            language_model_type=args.language_model_type,
            language_model_name=args.language_model_name,
            language_model_fine_tuning=args.language_model_fine_tuning,
            language_model_random_initialization=args.language_model_random_initialization,
            return_static_embeddings=args.return_static_embeddings,
            word_encoding_size=args.word_encoding_size,
            word_encoding_activation=args.word_encoding_activation,
            word_encoding_dropout=args.word_encoding_dropout,

            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            weight_decay=args.weight_decay,
            language_model_learning_rate=args.language_model_learning_rate,
            language_model_min_learning_rate=args.language_model_min_learning_rate,
            language_model_weight_decay=args.language_model_weight_decay
        )
    else:
        raise ValueError('Unsupported value for inventory: {}'.format(args.inventory))

    checkpoint_callback = ModelCheckpoint(
        monitor='val_roleset_f1',
        mode='max',
        filename='msrl-{val_roleset_f1:.4f}-{epoch:02d}',
        save_top_k=1,
    )

    trainer = Trainer.from_argparse_args(
        args,
        log_every_n_steps=100,
        flush_logs_every_n_steps=1000,
        callbacks=[checkpoint_callback])

    trainer.fit(model, data_module)
    
    data_module.save(trainer.log_dir)

    data_module.setup('test')
    trainer.test(model, datamodule=data_module)
