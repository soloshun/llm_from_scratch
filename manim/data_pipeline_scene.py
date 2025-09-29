"""
Manim scene for Part 3: The Data Pipeline

This scene provides a clean, minimalist visualization of how raw text
flows through our data pipeline to become training-ready batches.

To render this scene, run the following command:
(Activate your conda environment first)

For 1080p quality:
manim -pqh data_pipeline_scene.py DataPipelineScene

For 720p quality:
manim -pql data_pipeline_scene.py DataPipelineScene
"""

from manim import *
import numpy as np

class DataPipelineScene(Scene):
    def construct(self):
        # 1. Title
        title = Text("Data Pipeline", font_size=48).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # 2. Stage 1: Raw Text Input
        stage1_title = Text("Stage 1: Raw Text", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(stage1_title))
        
        # Show flowing text
        text_content = "I HAD always thought Jack..."
        flowing_text = Text(text_content, font_size=28).next_to(stage1_title, DOWN, buff=0.5)
        self.play(Write(flowing_text, run_time=2))
        self.wait(1)
        
        # 3. Stage 2: Tokenization
        self.play(FadeOut(stage1_title))
        stage2_title = Text("Stage 2: Tokenization", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(stage2_title))
        
        # Transform text into token boxes
        tokens = VGroup(
            *[Rectangle(width=0.8, height=0.5, fill_color=BLUE, fill_opacity=0.7).add(
                Text(str(token), font_size=20, color=WHITE)
            ) for token in [40, 367, 2885, 1464, 1807, 3619, 402, 271]]
        ).arrange(RIGHT, buff=0.1).next_to(stage2_title, DOWN, buff=0.5)
        
        self.play(
            Transform(flowing_text, tokens),
            run_time=2
        )
        self.wait(1)

        # 4. Stage 3: Sliding Window - Fixed positioning
        self.play(FadeOut(stage2_title))
        stage3_title = Text("Stage 3: Sliding Window", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(stage3_title))
        
        # Move tokens down to avoid overlap with title
        tokens_positioned = tokens.copy().next_to(stage3_title, DOWN, buff=1.0)
        self.play(Transform(flowing_text, tokens_positioned))
        
        # Create sliding window visualization
        window_size = 4
        window = Rectangle(
            width=window_size * 0.9, 
            height=0.7, 
            color=YELLOW, 
            stroke_width=3
        ).move_to(tokens_positioned[:window_size].get_center())
        
        self.play(Create(window))
        self.wait(0.5)
        
        # Show input-target pair creation - positioned to avoid overlap
        input_label = Text("Input", font_size=20, color=YELLOW).next_to(tokens_positioned, UP, buff=0.5)
        
        # Better target visualization with highlighted next token
        next_token_highlight = Rectangle(
            width=0.8, height=0.5, 
            color=GREEN, 
            stroke_width=4,
            fill_opacity=0
        ).move_to(tokens_positioned[window_size].get_center())
        
        target_label = Text("Target", font_size=20, color=GREEN).next_to(next_token_highlight, DOWN, buff=0.3)
        
        self.play(
            Write(input_label),
            Create(next_token_highlight),
            Write(target_label)
        )
        self.wait(1)
        
        # Animate sliding with better visual feedback
        for i in range(1, 4):  # Slide 3 times
            new_window = Rectangle(
                width=window_size * 0.9, 
                height=0.7, 
                color=YELLOW, 
                stroke_width=3
            ).move_to(tokens_positioned[i:i+window_size].get_center())
            
            new_next_token_highlight = Rectangle(
                width=0.8, height=0.5, 
                color=GREEN, 
                stroke_width=4,
                fill_opacity=0
            ).move_to(tokens_positioned[i+window_size].get_center())
            
            self.play(
                Transform(window, new_window),
                Transform(next_token_highlight, new_next_token_highlight),
                run_time=0.8
            )
            self.wait(0.5)

        # 5. Stage 4: Batching
        self.play(
            FadeOut(stage3_title), 
            FadeOut(window), 
            FadeOut(input_label),
            FadeOut(next_token_highlight), 
            FadeOut(target_label),
            FadeOut(flowing_text)
        )
        
        stage4_title = Text("Stage 4: Batching", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(stage4_title))
        
        # Show multiple input-target pairs being batched
        batch_rows = VGroup()
        colors = [BLUE, PURPLE, ORANGE, RED]
        
        for i, color in enumerate(colors):
            row = VGroup(
                *[Rectangle(width=0.6, height=0.4, fill_color=color, fill_opacity=0.6) 
                  for _ in range(4)]
            ).arrange(RIGHT, buff=0.05)
            batch_rows.add(row)
        
        batch_rows.arrange(DOWN, buff=0.1).next_to(stage4_title, DOWN, buff=0.5)
        
        # Animate rows appearing one by one
        for row in batch_rows:
            self.play(Create(row), run_time=0.5)
        
        # Add batch brackets
        left_bracket = Text("[", font_size=60).next_to(batch_rows, LEFT, buff=0.1)
        right_bracket = Text("]", font_size=60).next_to(batch_rows, RIGHT, buff=0.1)
        batch_label = Text("Batch", font_size=28, color=GREEN).next_to(batch_rows, DOWN, buff=0.3)
        
        self.play(
            Write(left_bracket),
            Write(right_bracket),
            Write(batch_label)
        )
        self.wait(2)  # Let batching complete fully

        # Complete Stage 4 fadeout first
        self.play(
            FadeOut(batch_rows),
            FadeOut(left_bracket),
            FadeOut(right_bracket),
            FadeOut(batch_label),
            FadeOut(stage4_title),
            run_time=1
        )
        self.wait(0.5)

        # 6. Final Stage: Ready for Model - Much more lively!
        final_title = Text("Ready for Training!", font_size=40, color=GREEN).next_to(title, DOWN, buff=1)
        self.play(Write(final_title))
        
        # Create model box
        model_box = Rectangle(
            width=3, height=2, 
            fill_color=GRAY, fill_opacity=0.3,
            stroke_color=WHITE
        ).next_to(final_title, DOWN, buff=1)
        model_text = Text("LLM Model", font_size=24).move_to(model_box.get_center())
        
        self.play(Create(model_box), Write(model_text))
        self.wait(0.5)
        
        # Animate batches flowing through the pipeline with different colors
        batch_colors = [BLUE, PURPLE, ORANGE, RED, GREEN]  # Different colors for each batch
        
        for batch_idx in range(3):  # Show 3 batches flowing through
            # Create a batch that moves with varied colors
            batch_color = batch_colors[batch_idx % len(batch_colors)]
            moving_batch = VGroup(
                *[Rectangle(width=0.4, height=0.3, fill_color=batch_color, fill_opacity=0.7) 
                  for _ in range(4)]
            ).arrange(RIGHT, buff=0.05).next_to(model_box, LEFT, buff=2)
            
            # Animate batch moving into model
            self.play(Create(moving_batch), run_time=0.5)
            self.play(
                moving_batch.animate.next_to(model_box, LEFT, buff=0.1),
                run_time=1
            )
            
            # Model "processes" the batch (flashing effect with batch color)
            for _ in range(2):
                self.play(
                    model_box.animate.set_fill(batch_color, opacity=0.4),
                    run_time=0.2
                )
                self.play(
                    model_box.animate.set_fill(GRAY, opacity=0.3),
                    run_time=0.2
                )
            
            # Batch disappears (absorbed into model)
            self.play(FadeOut(moving_batch), run_time=0.3)
            
            # Output appears and flies away (with corresponding color)
            output_token = Circle(radius=0.15, fill_color=batch_color, fill_opacity=0.8)
            output_token.move_to(model_box.get_right() + RIGHT * 0.2)
            self.play(Create(output_token), run_time=0.3)
            self.play(
                output_token.animate.shift(RIGHT * 2 + UP * 0.5),
                FadeOut(output_token),
                run_time=0.8
            )
        
        self.wait(1)

        # 7. Final summary
        summary = Text(
            "Raw Text → Tokens → Sliding Window → Batches → Model",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN, buff=0.5)
        
        self.play(Write(summary))
        self.wait(2)

        # 8. Final fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
