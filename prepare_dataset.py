"""
Dataset preparation utility for Informal-to-Formal text transformation.

This script downloads and prepares datasets from various sources:
1. GYAFC (Grammarly's Yahoo Answers Formality Corpus)
2. Hugging Face datasets
3. Custom datasets

Usage:
    python prepare_dataset.py --source gyafc --output ./data
    python prepare_dataset.py --source huggingface --dataset s-nlp/paradetox --output ./data
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
import requests
from datasets import load_dataset, Dataset, DatasetDict


def download_gyafc(output_dir: str = "./data/gyafc"):
    """
    Download GYAFC dataset from GitHub.
    
    Note: You may need to manually download from:
    https://github.com/raosudha89/GYAFC-corpus
    """
    print("Downloading GYAFC dataset...")
    print("Please manually download from: https://github.com/raosudha89/GYAFC-corpus")
    print(f"Extract to: {output_dir}")
    
    # Create directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    return output_dir


def load_from_huggingface(dataset_name: str = "jxm/informal_to_formal"):
    """
    Load informal-to-formal dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
    
    Returns:
        DatasetDict containing train/validation/test splits
    """
    print(f"Loading dataset from Hugging Face: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name)
        print(f"Dataset loaded successfully!")
        print(dataset)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to synthetic dataset...")
        return create_synthetic_dataset()


def create_synthetic_dataset(size: str = "medium") -> DatasetDict:
    """
    Create a synthetic informal-to-formal dataset.
    
    Args:
        size: Dataset size - "small" (50 pairs), "medium" (200 pairs), "large" (500+ pairs)
    
    Returns:
        DatasetDict with train/validation/test splits
    """
    print(f"Creating synthetic dataset (size: {size})...")
    
    # Base pairs
    base_pairs = [
        # Requests
        ("send me the file now", "Could you please send me the file at your earliest convenience?"),
        ("need this asap", "I would appreciate receiving this as soon as possible."),
        ("gimme a sec", "Please allow me a moment."),
        ("hey can u help?", "Hello, would you be able to assist me?"),
        ("get back to me", "Please respond at your convenience."),
        ("pls send the file", "Please send the file at your convenience."),
        ("need help with this", "I would appreciate assistance with this matter."),
        ("can u check this?", "Could you please review this?"),
        ("fix it now", "Please address this issue at your earliest convenience."),
        ("call me", "Please contact me at your convenience."),
        
        # Questions
        ("why didn't you finish the report?", "I noticed the report is incomplete. Could you please provide an update on its status?"),
        ("what's taking so long?", "May I inquire about the current progress?"),
        ("where's the data?", "Could you please direct me to the location of the data?"),
        ("did u see my email?", "Have you had the opportunity to review my email?"),
        ("when can we meet?", "When would be a convenient time for us to schedule a meeting?"),
        ("what's up with that?", "Could you please clarify the situation?"),
        ("what's the deal?", "What is the current situation?"),
        
        # Statements
        ("this is wrong", "I believe there may be an error in this."),
        ("that won't work", "I'm concerned that approach may not be effective."),
        ("u made a mistake", "It appears there may have been an oversight."),
        ("i'm busy right now", "I am currently occupied with other tasks."),
        ("can't do it today", "Unfortunately, I will not be able to complete this today."),
        ("i dunno", "I am uncertain about that."),
        ("gonna be late", "I will be arriving later than expected."),
        ("can't make it", "Unfortunately, I will be unable to attend."),
        ("i'm not gonna do that", "I will not be able to proceed with that request."),
        ("kinda busy", "I am somewhat occupied at the moment."),
        
        # Feedback
        ("good job", "Excellent work on this project."),
        ("looks ok to me", "This appears to be satisfactory."),
        ("not bad", "This is quite good."),
        ("yeah that's fine", "Yes, that would be acceptable."),
        ("nah i don't think so", "I respectfully disagree with that assessment."),
        ("awesome job", "You have done exceptional work."),
        ("really cool", "This is quite impressive."),
        
        # Short responses
        ("nope", "No, thank you."),
        ("yup", "Yes, certainly."),
        ("got it", "I understand completely."),
        ("my bad", "I apologize for that error."),
        ("no way", "I find that difficult to believe."),
        ("for sure", "Absolutely, I agree."),
        ("hang on", "Please wait a moment."),
        ("np", "You are welcome."),
        ("thx", "Thank you."),
        ("brb", "I will return shortly."),
        ("gtg", "I must leave now."),
        
        # Expressions
        ("whatever", "I understand your perspective."),
        ("who cares", "That may not be a primary concern."),
        ("let's do this", "Shall we proceed with this plan?"),
        ("cut it out", "Please discontinue that behavior."),
        ("knock it off", "Please cease that action."),
        ("chill out", "Please remain calm."),
        ("no biggie", "That is not a significant concern."),
        ("sorta like that", "It is somewhat similar to that."),
        ("tons of work", "I have a substantial amount of work."),
        ("super important", "This is extremely important."),
        ("that sucks", "That is unfortunate."),
        ("talk later", "Let us continue this conversation at a later time."),
        ("lemme know", "Please inform me when you have an update."),
        ("omg that's bad", "That is quite concerning."),
    ]
    
    # Add more variations for medium and large datasets
    if size in ["medium", "large"]:
        variations = [
            ("send it over", "Please send that to me."),
            ("need it now", "I require this urgently."),
            ("help me out", "Could you assist me, please?"),
            ("what's going on?", "Could you explain the situation?"),
            ("tell me more", "Could you please provide additional details?"),
            ("that's crazy", "That is quite remarkable."),
            ("pretty good", "This is satisfactory."),
            ("kinda weird", "This is somewhat unusual."),
            ("sounds good", "That sounds acceptable."),
            ("makes sense", "That is logical."),
            ("i see", "I understand."),
            ("ok cool", "Very well, thank you."),
            ("will do", "I will complete that."),
            ("on it", "I am working on that now."),
            ("hold on", "Please wait momentarily."),
            ("give me a minute", "Please allow me a moment."),
            ("catch you later", "I will speak with you later."),
            ("take care", "Best regards."),
            ("see ya", "Goodbye."),
            ("later", "Until next time."),
            ("thanks a bunch", "Thank you very much."),
            ("thanks a lot", "I greatly appreciate that."),
            ("no problem", "You are welcome."),
            ("sure thing", "Certainly, I will do that."),
            ("you bet", "Absolutely."),
            ("alright", "Very well."),
            ("fine by me", "That is acceptable to me."),
            ("i'm down", "I agree with that plan."),
            ("count me in", "I would like to participate."),
            ("not feeling it", "I do not think that is appropriate."),
            ("pass", "I decline."),
            ("maybe later", "Perhaps at a future time."),
            ("not right now", "Not at this moment."),
            ("in a bit", "Shortly."),
            ("almost done", "I am nearly finished."),
            ("working on it", "I am currently addressing that."),
            ("looking into it", "I am investigating that matter."),
            ("checking", "I am verifying that."),
            ("lemme think", "Allow me to consider that."),
            ("gimme a break", "Please be reasonable."),
            ("seriously?", "Are you certain?"),
            ("you kidding?", "Is that correct?"),
            ("no kidding", "That is accurate."),
            ("totally", "Completely."),
            ("absolutely", "Certainly."),
            ("definitely", "Without question."),
            ("probably", "Most likely."),
            ("maybe", "Perhaps."),
            ("doubt it", "I am skeptical."),
        ]
        base_pairs.extend(variations)
    
    if size == "large":
        more_variations = [
            ("what do you think?", "What is your opinion on this matter?"),
            ("let me know asap", "Please inform me as soon as possible."),
            ("gotta run", "I must depart now."),
            ("catch up soon", "Let us reconnect in the near future."),
            ("miss you", "I look forward to seeing you again."),
            ("good to see you", "It is pleasant to see you."),
            ("nice work", "You have performed admirably."),
            ("keep it up", "Continue your excellent work."),
            ("well done", "You have succeeded admirably."),
            ("congrats", "Congratulations on your achievement."),
            ("sorry about that", "I apologize for that matter."),
            ("my mistake", "That was my error."),
            ("oops", "I apologize for that oversight."),
            ("uh oh", "There appears to be an issue."),
            ("wow", "That is impressive."),
            ("amazing", "That is remarkable."),
            ("incredible", "That is extraordinary."),
            ("unbelievable", "That is difficult to comprehend."),
            ("no clue", "I have no information about that."),
            ("beats me", "I do not know."),
            ("who knows", "That is uncertain."),
            ("hard to say", "That is difficult to determine."),
            ("not sure", "I am uncertain."),
            ("could be", "That is possible."),
            ("might be", "That may be the case."),
            ("i guess", "I suppose so."),
            ("fair enough", "That is reasonable."),
            ("makes sense", "That is logical."),
            ("i suppose", "I believe that is accurate."),
            ("if you say so", "I defer to your judgment."),
            ("up to you", "The decision is yours."),
            ("your call", "You may decide."),
            ("whatever you want", "As you prefer."),
            ("doesn't matter", "That is not significant."),
            ("don't care", "I have no preference."),
            ("all good", "Everything is satisfactory."),
            ("no worries", "There is no concern."),
            ("it's fine", "That is acceptable."),
            ("it's ok", "That is satisfactory."),
            ("don't sweat it", "Do not be concerned."),
            ("take it easy", "Please relax."),
            ("calm down", "Please remain composed."),
            ("relax", "Please be calm."),
            ("just saying", "I am merely observing."),
            ("fyi", "For your information."),
            ("btw", "Incidentally."),
            ("anyways", "In any case."),
            ("anyhow", "Nevertheless."),
            ("so yeah", "Therefore."),
            ("basically", "Essentially."),
            ("pretty much", "Approximately."),
        ]
        base_pairs.extend(more_variations)
    
    # Create dataset splits
    import random
    random.seed(42)
    random.shuffle(base_pairs)
    
    total = len(base_pairs)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    train_pairs = base_pairs[:train_size]
    val_pairs = base_pairs[train_size:train_size + val_size]
    test_pairs = base_pairs[train_size + val_size:]
    
    def create_dataset_split(pairs):
        return Dataset.from_dict({
            "informal": [p[0] for p in pairs],
            "formal": [p[1] for p in pairs]
        })
    
    dataset = DatasetDict({
        "train": create_dataset_split(train_pairs),
        "validation": create_dataset_split(val_pairs),
        "test": create_dataset_split(test_pairs)
    })
    
    print(f"Dataset created with {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test samples")
    
    return dataset


def save_dataset(dataset: DatasetDict, output_dir: str = "./data/informal_formal"):
    """Save dataset to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as JSON for easy inspection
    for split in ["train", "validation", "test"]:
        output_file = Path(output_dir) / f"{split}.json"
        data = []
        for item in dataset[split]:
            data.append({
                "informal": item["informal"],
                "formal": item["formal"]
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {split} split to {output_file}")
    
    # Also save as Hugging Face dataset
    dataset.save_to_disk(output_dir + "_hf")
    print(f"Saved Hugging Face dataset to {output_dir}_hf")


def load_custom_dataset(input_file: str) -> DatasetDict:
    """
    Load custom dataset from JSON file.
    
    Expected format:
    [
        {"informal": "...", "formal": "..."},
        ...
    ]
    """
    print(f"Loading custom dataset from {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Split into train/val/test
    import random
    random.seed(42)
    random.shuffle(data)
    
    total = len(data)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    def create_split(subset):
        return Dataset.from_dict({
            "informal": [item["informal"] for item in subset],
            "formal": [item["formal"] for item in subset]
        })
    
    dataset = DatasetDict({
        "train": create_split(data[:train_size]),
        "validation": create_split(data[train_size:train_size + val_size]),
        "test": create_split(data[train_size + val_size:])
    })
    
    print(f"Loaded {len(dataset['train'])} train, {len(dataset['validation'])} validation, {len(dataset['test'])} test samples")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare informal-to-formal dataset")
    parser.add_argument(
        "--source",
        type=str,
        choices=["synthetic", "huggingface", "gyafc", "custom"],
        default="synthetic",
        help="Dataset source"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jxm/informal_to_formal",
        help="Dataset name for HuggingFace source"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file path for custom dataset"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "medium", "large"],
        default="large",
        help="Size of synthetic dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/informal_formal",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load dataset based on source
    if args.source == "synthetic":
        dataset = create_synthetic_dataset(size=args.size)
    elif args.source == "huggingface":
        dataset = load_from_huggingface(args.dataset)
    elif args.source == "gyafc":
        output_dir = download_gyafc(args.output)
        print(f"Please manually process GYAFC data from {output_dir}")
        return
    elif args.source == "custom":
        if not args.input:
            print("Error: --input required for custom dataset")
            return
        dataset = load_custom_dataset(args.input)
    
    # Save dataset
    save_dataset(dataset, args.output)
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"Dataset saved to: {args.output}")
    print(f"Total samples: {len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])}")
    print(f"  Train: {len(dataset['train'])}")
    print(f"  Validation: {len(dataset['validation'])}")
    print(f"  Test: {len(dataset['test'])}")
    print("\nYou can now use this dataset in train_model.ipynb")


if __name__ == "__main__":
    main()
