\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
   \relative c'{\time 3/4 s2 g'4 | c c d | e2 g,4 | c2 e4 | d2 g,4 | d'2 e4 | f d b | g a b | c2 g4 | c2 d4 | e e g, | c2 e4 | d2. | d2 e4 | f d b | g a b | c2 s4 \bar "|."}

}
\layout {
  \context {
      \Score 
      proportionalNotationDuration =  #(ly:make-moment 1/5)
 }
 }
\midi {\tempo 4 = 60}
}
