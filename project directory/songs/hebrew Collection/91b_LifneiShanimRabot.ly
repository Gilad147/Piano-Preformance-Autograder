\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  \key f \major \relative c'{\numericTimeSignature f f8 g a4 a8 bes | c4 d8 c a2 | c4 bes8 a g2 | bes4 a8 g f2 | f4 f8 g a4 a8 bes | c4 d8 c a2 | c4 bes8 a g4 a8 g | f1 | c'4 bes8 a g4 c,8 c | bes'4 a8 g f2 | c'4 bes8 a g4 c,8 c | bes'4 a8 g f2 | f4 f8 g a4 a8 bes | c4 d8 c a2 | c4 bes8 a g4 a8 g f1 \bar "|."}

}
\layout {
  \context {
      \Score 
      proportionalNotationDuration =  #(ly:make-moment 1/5)
 }
 }
\midi {\tempo 4 = 60}
}
