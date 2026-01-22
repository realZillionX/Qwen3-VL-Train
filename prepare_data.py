import json
import os
import argparse
from pathlib import Path

# Task -> Prompt Mapping for Eyeballing
EYEBALLING_PROMPTS = {
    # Point Tasks
    "circle_center": "Which point looks like the center of the circle? Choose the best option.",
    "circumcenter": "Which point looked like the circumcenter of the triangle? Choose the best option.",
    "fermat_point": "Which point looks like the Fermat point of the triangle? Choose the best option.",
    "incenter": "Which point looks like the incenter of the triangle? Choose the best option.",
    "midpoint": "Which point looks like the midpoint of the segment? Choose the best option.",
    "orthocenter": "Which point looks like the orthocenter of the triangle? Choose the best option.",
    "point_reflection": "Which point looks like the reflection of the source point? Choose the best option.",
    "ray_intersection": "Which point looks like the intersection of the rays? Choose the best option.",
    "triangle_center": "Which point looks like the center of the triangle? Choose the best option.",
    
    # Line Tasks
    "angle_bisector": "Which line looks like the angle bisector? Choose the best option.",
    "arc_connect": "Which line correctly connects the arcs? Choose the best option.",
    "circle_tangent_line": "Which line looks tangent to the circle? Choose the best option.",
    "circle_tangent_point": "Which point looks like the point of tangency? Choose the best option.",
    "parallel": "Which line looks parallel to the reference line? Choose the best option.",
    "perpendicular": "Which line looks perpendicular to the reference line? Choose the best option.",
    "perpendicular_bisector": "Which line looks like the perpendicular bisector? Choose the best option.",
    "ray_reflect": "Which ray looks like the correct reflection? Choose the best option.",
    
    # Shape Tasks
    "isosceles_trapezoid": "Which point completes the isosceles trapezoid? Choose the best option.",
    "parallelogram": "Which point completes the parallelogram? Choose the best option.",
    "right_triangle": "Which point forms a right triangle? Choose the best option.",
    "square_outlier": "Which point is the outlier that does not fit the square pattern? Choose the best option."
}

# Fixed Prompt for Maze
MAZE_PROMPT = "Find a path connecting two red dots without touching the black walls in the maze. Movement is between adjacent cells through shared edges only (no diagonal corner moves). Each cell has its ID printed on it. Present your answer as a list of cell IDs. Example: [1, 4, 3, 2]. Must answer now without asking for clarifications."

def get_eyeballing_prompt(task_type, mode='sft'):
    base_prompt = EYEBALLING_PROMPTS.get(task_type, "Select the correct option from the image.")
    
    if mode == 'sft':
        suffix = "\n不要输出思考过程。\nPlease output your final answer within <answer>...</answer> tags."
    else: # grpo
        suffix = "\n输出思考过程，并把答案用<answer></answer>包裹。"
        
    return base_prompt + suffix

def get_maze_prompt(mode='sft'):
    base_prompt = MAZE_PROMPT
    
    if mode == 'sft':
        suffix = "\n不要输出思考过程。\nPlease output your final answer within <answer>...</answer> tags."
    else: # grpo
        suffix = "\n输出思考过程，并把答案用<answer></answer>包裹。"
        
    return base_prompt + suffix

def process_eyeballing(data_root, mode):
    dataset_dir = os.path.join(data_root, 'dataset_eyeballing')
    json_path = os.path.join(dataset_dir, 'data.json')
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping eyeballing.")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        image_path = os.path.join(dataset_dir, item['image'])
        task_type = item.get('task_type', '')
        
        prompt = get_eyeballing_prompt(task_type, mode)
        
        entry = {
            "query": prompt,
            "response": f"<answer>{item['correct_option']}</answer>",
            "images": [image_path],
            "solution": item['correct_option']
        }
        processed.append(entry)
    
    return processed

def process_maze(data_root, mode):
    dataset_dir = os.path.join(data_root, 'dataset_maze')
    json_path = os.path.join(dataset_dir, 'data.json')
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping maze.")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        image_path = os.path.join(dataset_dir, item['image'])
        
        solution_str = json.dumps(item['solution_path_cell_ids'])
        prompt = get_maze_prompt(mode)
        
        entry = {
            "query": prompt,
            "response": f"<answer>{solution_str}</answer>",
            "images": [image_path],
            "solution": solution_str
        }
        processed.append(entry)
    
    return processed

def main():
    parser = argparse.ArgumentParser(description="Convert VLMPuzzle datasets to ms-swift JSONL format")
    parser.add_argument("--data_root", type=str, default="/inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset", help="Root directory")
    
    args = parser.parse_args()
    
    # Generate SFT Data
    sft_data = []
    sft_data.extend(process_eyeballing(args.data_root, mode='sft'))
    sft_data.extend(process_maze(args.data_root, mode='sft'))
    
    sft_out = "train_sft.jsonl"
    with open(sft_out, 'w') as f:
        for entry in sft_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(sft_data)} samples to {sft_out}")
    
    # Generate GRPO Data
    grpo_data = []
    grpo_data.extend(process_eyeballing(args.data_root, mode='grpo'))
    grpo_data.extend(process_maze(args.data_root, mode='grpo'))
    
    grpo_out = "train_grpo.jsonl"
    with open(grpo_out, 'w') as f:
        for entry in grpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(grpo_data)} samples to {grpo_out}")

if __name__ == "__main__":
    main()
