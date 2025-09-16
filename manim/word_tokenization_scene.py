"""
Manim scene for Part 2: Word-Based Tokenization

This scene explains the concept of word-based tokenization and
the Out-of-Vocabulary (OOV) problem, then shows how special
tokens like <unk> provide a solution.

To render this scene, run the following command:
(Activate your conda environment first)

For 1080p quality:
manim -pqh word_tokenization_scene.py WordTokenizationScene

For 720p quality:
manim -pql word_tokenization_scene.py WordTokenizationScene
"""

from manim import *

class WordTokenizationScene(Scene):
    def construct(self):
        # 1. Title
        title = Text("Word-Based Tokenization", font_size=48).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # 2. Show a simple sentence
        sentence = Text("My cat is fluffy.", font_size=36)
        self.play(Write(sentence))
        self.wait(1)

        # 3. Animate splitting into tokens
        tokens = VGroup(
            Text("My"), Text("cat"), Text("is"), Text("fluffy"), Text(".")
        ).arrange(RIGHT, buff=0.5).next_to(sentence, DOWN, buff=1)
        
        token_boxes = VGroup(*[SurroundingRectangle(t, buff=0.2, corner_radius=0.1) for t in tokens])

        self.play(TransformFromCopy(sentence, tokens))
        self.play(Create(token_boxes))
        self.wait(1)

        # 4. Introduce the concept of a Vocabulary
        vocab_title = Text("Vocabulary", font_size=32).to_edge(LEFT, buff=0.5).shift(UP*1.5)
        vocab_box = SurroundingRectangle(vocab_title, buff=0.3, corner_radius=0.1, color=YELLOW)
        vocab_map = VGroup(
            Text("'My' -> 0", font_size=24),
            Text("'cat' -> 1", font_size=24),
            Text("'is' -> 2", font_size=24),
            Text("'fluffy' -> 3", font_size=24),
            Text("'.' -> 4", font_size=24)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(vocab_box, DOWN, buff=0.3)
        
        self.play(Write(vocab_title), Create(vocab_box))
        self.play(Write(vocab_map))
        self.wait(2)

        # Define the vocab dictionary for programmatic lookup
        vocab = { "My": 0, "cat": 1, "is": 2, "fluffy": 3, ".": 4 }

        # 5. Show mapping tokens to IDs (with corrected alignment)
        mapped_ids = VGroup()
        for token in tokens:
            # Position each ID directly below its token
            id_text = Text(str(vocab[token.text]), font_size=36).next_to(token, DOWN, buff=1.0)
            mapped_ids.add(id_text)

        arrows = VGroup(*[
            Arrow(token.get_bottom(), mapped_id.get_top(), buff=0.1)
            for token, mapped_id in zip(tokens, mapped_ids)
        ])
        
        self.play(
            *[TransformFromCopy(t, i) for t, i in zip(tokens, mapped_ids)],
            *[Create(arrow) for arrow in arrows]
        )
        self.wait(2)
        
        # 6. Introduce the Out-of-Vocabulary (OOV) problem
        self.play(
            FadeOut(sentence), FadeOut(tokens), FadeOut(token_boxes), 
            FadeOut(mapped_ids), FadeOut(arrows)
        )
        
        oov_title = Text("The Problem: Unknown Words", color=RED, font_size=40).move_to(ORIGIN).shift(UP*2)
        self.play(Write(oov_title))
        
        new_sentence = Text("My dog is sleeping.", font_size=36).next_to(oov_title, DOWN, buff=1)
        self.play(Write(new_sentence))
        self.wait(1)
        
        # 7. Animate trying to tokenize the new sentence
        new_tokens = VGroup(
            Text("My"), Text("dog", color=RED), Text("is"), Text("sleeping", color=RED), Text(".")
        ).arrange(RIGHT, buff=0.5).next_to(new_sentence, DOWN, buff=1)
        
        q_marks = VGroup(
            Text("?", color=RED, font_size=48).next_to(new_tokens[1], DOWN, buff=0.5),
            Text("?", color=RED, font_size=48).next_to(new_tokens[3], DOWN, buff=0.5)
        )
        
        self.play(TransformFromCopy(new_sentence, new_tokens))
        self.wait(1)
        self.play(
            Indicate(vocab_map[1], scale_factor=1.2, color=GREEN), # 'cat'
            Indicate(vocab_map[2], scale_factor=1.2, color=GREEN)  # 'is'
        )
        self.play(Write(q_marks))
        self.wait(2)

        # 8. Introduce the <unk> token solution
        self.play(
            FadeOut(oov_title), FadeOut(new_sentence), 
            FadeOut(new_tokens), FadeOut(q_marks)
        )
        
        solution_title = Text("The Solution: <|unk|> Token", color=GREEN, font_size=40).move_to(ORIGIN).shift(UP*2)
        self.play(Write(solution_title))
        
        unk_token = Text("'<|unk|>' -> 5", font_size=24).next_to(vocab_map, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(Write(unk_token))
        self.wait(1)

        # 9. Re-tokenize with the <unk> solution
        final_tokens = VGroup(
            Text("My"), Text("dog"), Text("is"), Text("sleeping"), Text(".")
        ).arrange(RIGHT, buff=0.5).next_to(solution_title, DOWN, buff=1)
        
        final_ids = VGroup(
            Text("0"), Text("5", color=GREEN), Text("2"), Text("5", color=GREEN), Text("4")
        )

        # Correctly position each ID below its token
        for i, token in enumerate(final_tokens):
            final_ids[i].next_to(token, DOWN, buff=1.0)

        final_arrows = VGroup(*[
            Arrow(token.get_bottom(), mapped_id.get_top(), buff=0.1)
            for token, mapped_id in zip(final_tokens, final_ids)
        ])

        self.play(Write(final_tokens))
        self.wait(1)
        self.play(
            Transform(final_tokens[1], final_ids[1]),
            Transform(final_tokens[3], final_ids[3]),
            TransformFromCopy(final_tokens[0], final_ids[0]),
            TransformFromCopy(final_tokens[2], final_ids[2]),
            TransformFromCopy(final_tokens[4], final_ids[4]),
            Create(final_arrows)
        )
        self.wait(3)

        # 10. Final fade out
        self.play(
            FadeOut(title), FadeOut(solution_title), FadeOut(final_tokens),
            FadeOut(final_ids), FadeOut(final_arrows),
            FadeOut(vocab_title), FadeOut(vocab_box), FadeOut(vocab_map), FadeOut(unk_token)
        )
        self.wait(1)
