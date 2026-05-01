# Data Directory

## Do NOT commit raw or processed images to Git.

## Expected structure after dataset setup

```
data/
├── processed/
│   ├── train/
│   │   ├── Biliary_Leaks/   (110 images)
│   │   ├── Lithiasis/       (505 images)
│   │   ├── Normal/          (197 images)
│   │   └── Stricture/       (255 images)
│   ├── val/
│   │   ├── Biliary_Leaks/   (24 images)
│   │   ├── Lithiasis/       (98 images)
│   │   ├── Normal/          (59 images)
│   │   └── Stricture/       (53 images)
│   └── test/
│       ├── Biliary_Leaks/   (17 images)
│       ├── Lithiasis/       (123 images)
│       ├── Normal/          (43 images)
│       └── Stricture/       (84 images)
└── README.md
```

## Dataset source

- Article: https://doi.org/10.6084/m9.figshare.31079236
- Download: https://figshare.com/ndownloader/files/61063177
- GitHub reference: https://github.com/monicaccmartins/MIQR-CC-Dataset

See SETUP.md in the project root for full setup instructions.
