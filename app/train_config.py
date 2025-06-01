import os
import sys
sys.path.append('/app/ai-toolkit')
from toolkit.job import run_job
from collections import OrderedDict

train_config = OrderedDict([
    ('job', 'extension'),
    ('config', OrderedDict([
        # this name will be the folder and filename name
        ('name', 'my_first_flux_lora_v1'),
        ('process', [
            OrderedDict([
                ('type', 'sd_trainer'),
                # root folder to save training sessions/samples/weights
                ('training_folder', '/workspace/output1'),
                # uncomment to see performance stats in the terminal every N steps
                ('performance_log_every', 100),
                ('device', 'cuda:0'),
                # if a trigger word is specified, it will be added to captions of training data if it does not already exist
                # alternatively, in your captions you can add [trigger] and it will be replaced with the trigger word
                ('trigger_word', 'woman'),
                ('network', OrderedDict([
                    ('type', 'lora'),
                    ('linear', 128),
                    ('linear_alpha', 128),
                    ('network_kwargs', OrderedDict([
                        ('only_if_contains', [
                            "transformer.single_transformer_blocks.9.",
                            "transformer.single_transformer_blocks.25."
                        ])
                    ]))
                ])),
                ('save', OrderedDict([
                    ("dtype", "bfloat16"),
                    ('save_every', 250),  # save every this many steps
                    ('max_step_saves_to_keep', 4)  # how many intermittent saves to keep
                ])),
                ('datasets', [
                    # datasets are a folder of images. captions need to be txt files with the same name as the image
                    # for instance image2.jpg and image2.txt. Only jpg, jpeg, and png are supported currently
                    # images will automatically be resized and bucketed into the resolution specified
                    OrderedDict([
                        ('folder_path', '/app/input'),
                        ('caption_ext', 'txt'),
                        ('caption_dropout_rate', 0.05),  # will drop out the caption 5% of time
                        ('shuffle_tokens', False),  # shuffle caption order, split by commas
                        ('cache_latents_to_disk', True),  # leave this true unless you know what you're doing
                        ('num_workers', 8),
                        ('pin_memory', True),
                        #('resolution', [1024])
                        ('resolution', [512, 768, 1024])  # flux enjoys multiple resolutions
                    ])
                ]),
                ('train', OrderedDict([
                    ('batch_size', 8),
                    ('steps', 200),  # total number of steps to train 500 - 4000 is a good range
                    ('gradient_accumulation_steps', 1),
                    ('train_unet', True),
                    ('train_text_encoder', False),  # probably won't work with flux
                    ('content_or_style', 'balanced'),  # content, style, balanced
                    ('gradient_checkpointing', True),  # need the on unless you have a ton of vram
                    #('gradient_checkpointing', False),
                    ('noise_scheduler', 'flowmatch'),  # for training only
                    ('optimizer', 'adamw8bit'),
                    ('lr', 5e-4),

                    # uncomment this to skip the pre training sample
                    ('skip_first_sample', True),

                    # uncomment to completely disable sampling
                    ('disable_sampling', False),

                    # uncomment to use new vell curved weighting. Experimental but may produce better results
                    ('linear_timesteps', True),

                    # ema will smooth out learning, but could slow it down. Recommended to leave on.
                    #('ema_config', OrderedDict([
                    #    ('use_ema', True),
                    #    ('ema_decay', 0.99)
                    # ])),

                    # will probably need this if gpu supports it for flux, other dtypes may not work correctly
                    ('dtype', 'bfloat16')
                ])),
                ('model', OrderedDict([
                    # huggingface model name or path
                    ('name_or_path', '/workspace/flux1-dev'),
                    ('is_flux', True),
                    ('quantize', True),  # run 8bit mixed precision
                    ('low_vram', False),  # uncomment this if the GPU is connected to your monitors. It will use less vram to quantize, but is slower.
                ])),
                ('sample', OrderedDict([
                    ('sampler', 'flowmatch'),  # must match train.noise_scheduler
                    ('sample_every', 50),  # sample every this many steps
                    ('width', 512),
                    ('height', 512),
                    ('prompts', [
                        # you can add [trigger] to the prompts here and it will be replaced with the trigger word
                        #'[trigger] holding a sign that says \'I LOVE PROMPTS!\'',
                        'woman with red hair, playing chess at the park, bomb going off in the background',
                    ]),
                    ('neg', ''),  # not used on flux
                    ('seed', 42),
                    ('walk_seed', True),
                    ('guidance_scale', 4),
                    ('sample_steps', 20)
                ]))
            ])
        ])
    ])),
    # you can add any additional meta info here. [name] is replaced with config name at top
    ('meta', OrderedDict([
        ('name', '[name]'),
        ('version', '1.0')
    ]))
])
