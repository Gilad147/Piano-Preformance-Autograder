\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  \relative c'{\numericTimeSignature c4 d4 e4 c4 d4 e4 f4 d4 e4 f4 g4 e4 d1 c4 d4 e4 c4 d4 e4 f4 d4 e4 f4 g4 e4 c1 \bar "|."}

}
\layout {
  \context {
      \Score 
      proportionalNotationDuration =  #(ly:make-moment 1/5)
 }
 }
\midi {\tempo 4 = 60}
}
