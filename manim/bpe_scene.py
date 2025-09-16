"""
Manim scene for Part 2: Byte Pair Encoding (BPE)

This scene provides a simplified, step-by-step visualization
of how the Byte Pair Encoding (BPE) algorithm works. It starts
with a small corpus, builds a vocabulary by merging frequent
pairs, and shows how BPE can tokenize unknown words.

To render this scene, run the following command:
(Activate your conda environment first)

For 1080p quality:
manim -pqh bpe_scene.py BPEScene

For 720p quality:
manim -pql bpe_scene.py BPEScene
"""

from manim import *

class BPEScene(Scene):
    def construct(self):
        # 1. Title
        title = Text("Byte Pair Encoding (BPE)", font_size=48).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # 2. Position Corpus and Vocabulary areas
        corpus_title = Text("Corpus:", font_size=32)
        corpus_text = VGroup(
            Text("low", font_size=28),
            Text("low", font_size=28),
            Text("lowest", font_size=28),
            Text("newer", font_size=28),
            Text("wider", font_size=28),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(corpus_title, DOWN, buff=0.3)
        corpus_group = VGroup(corpus_title, corpus_text).to_edge(LEFT, buff=0.8)

        vocab_title = Text("Vocabulary:", font_size=32)
        vocab_text = Text("l, o, w, e, s, t, n, r, d", font_size=28)
        vocab_group = VGroup(vocab_title, vocab_text).arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_corner(DR, buff=0.8)

        self.play(Write(corpus_group))
        self.wait(1)
        
        # 3. Step 1: Character Vocabulary
        step1_title = Text("Step 1: Start with characters", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(step1_title))
        self.play(Write(vocab_group))
        self.wait(2)
        
        # 4. Step 2: Merge 'l' + 'o' -> 'lo'
        self.play(FadeOut(step1_title))
        step2_title = Text("Step 2: Merge most frequent pair ('lo')", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(step2_title))
        
        merge_rule = Text("'l' + 'o' -> 'lo'", font_size=32, color=YELLOW).next_to(step2_title, DOWN, buff=0.7)
        self.play(Write(merge_rule))
        
        new_vocab_text = Text("..., 'lo'", font_size=28, t2c={"'lo'": YELLOW}).next_to(vocab_title, DOWN, buff=0.3)
        new_corpus_text = VGroup(
            Text("lo w", font_size=28, t2c={"lo": YELLOW}),
            Text("lo w", font_size=28, t2c={"lo": YELLOW}),
            Text("lo west", font_size=28, t2c={"lo": YELLOW}),
            Text("newer", font_size=28),
            Text("wider", font_size=28),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(corpus_title, DOWN, buff=0.3)

        self.play(Transform(vocab_text, new_vocab_text), Transform(corpus_text, new_corpus_text))
        self.wait(2)

        # 5. Step 3: Merge 'e' + 'r' -> 'er'
        self.play(FadeOut(step2_title), FadeOut(merge_rule))
        step3_title = Text("Step 3: Merge next frequent pair ('er')", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(step3_title))
        
        merge_rule2 = Text("'e' + 'r' -> 'er'", font_size=32, color=GREEN).next_to(step3_title, DOWN, buff=0.7)
        self.play(Write(merge_rule2))
        
        new_vocab_text2 = Text("... 'lo', 'er'", font_size=28, t2c={"'lo'": YELLOW, "'er'": GREEN}).next_to(vocab_title, DOWN, buff=0.3)
        new_corpus_text2 = VGroup(
            Text("lo w", font_size=28, t2c={"lo": YELLOW}),
            Text("lo w", font_size=28, t2c={"lo": YELLOW}),
            Text("lo west", font_size=28, t2c={"lo": YELLOW}),
            Text("new er", font_size=28, t2c={"er": GREEN}),
            Text("wid er", font_size=28, t2c={"er": GREEN}),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(corpus_title, DOWN, buff=0.3)

        self.play(Transform(vocab_text, new_vocab_text2), Transform(corpus_text, new_corpus_text2))
        self.wait(2)

        # 6. Step 4: Merge 'lo' + 'w' -> 'low'
        self.play(FadeOut(step3_title), FadeOut(merge_rule2))
        step4_title = Text("Step 4: Merge again ('low')", font_size=36).next_to(title, DOWN, buff=0.7)
        self.play(Write(step4_title))
        
        merge_rule3 = Text("'lo' + 'w' -> 'low'", font_size=32, color=ORANGE).next_to(step4_title, DOWN, buff=0.7)
        self.play(Write(merge_rule3))

        new_vocab_text3 = Text("... 'er', 'low'", font_size=28, t2c={"'er'": GREEN, "'low'": ORANGE}).next_to(vocab_title, DOWN, buff=0.3)
        new_corpus_text3 = VGroup(
            Text("low", font_size=28, t2c={"low": ORANGE}),
            Text("low", font_size=28, t2c={"low": ORANGE}),
            Text("low est", font_size=28, t2c={"low": ORANGE}),
            Text("new er", font_size=28, t2c={"er": GREEN}),
            Text("wid er", font_size=28, t2c={"er": GREEN}),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(corpus_title, DOWN, buff=0.3)

        self.play(Transform(vocab_text, new_vocab_text3), Transform(corpus_text, new_corpus_text3))
        self.wait(2)
        
        # 7. Final Tokenization
        self.play(
            FadeOut(step4_title), FadeOut(merge_rule3), FadeOut(corpus_group), FadeOut(vocab_group)
        )
        
        final_title = Text("Tokenizing a New Word", font_size=40).next_to(title, DOWN, buff=0.7)
        self.play(Write(final_title))
        
        final_vocab = Text("Final Vocabulary includes: 'er', 'low', ...", font_size=28).next_to(final_title, DOWN, buff=0.5)
        self.play(Write(final_vocab))

        new_word = Text("slower", font_size=36).move_to(ORIGIN)
        self.play(Write(new_word))
        self.wait(1)
        
        tokenized_word = VGroup(
            Text("s"), Text("low", color=ORANGE), Text("er", color=GREEN)
        ).arrange(RIGHT, buff=0.4).next_to(new_word, DOWN, buff=0.8)
        
        token_boxes = VGroup(*[SurroundingRectangle(t, buff=0.2, color=BLUE, corner_radius=0.1) for t in tokenized_word])
        
        self.play(TransformFromCopy(new_word, tokenized_word))
        self.play(Create(token_boxes))
        
        insight = Text("BPE tokenizes unknown words into known subwords!", font_size=28, color=YELLOW).next_to(tokenized_word, DOWN, buff=1)
        self.play(Write(insight))
        self.wait(3)

        # 8. Final fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
