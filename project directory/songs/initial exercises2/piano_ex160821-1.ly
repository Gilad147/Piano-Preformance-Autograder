\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % piano adventures, premier level. p 47

\new PianoStaff
<<
\new Staff="upper"
\relative c' {
  \clef treble
  \key c \major
  \time 3/4

	R2. | r2 d4 | e e d | e2 f4 | g4 g r | R2. | f4 e d | R2. \bar "|."
}
\new Staff="lower"
\relative c {
  \clef bass
  \key c \major
  \time 3/4

   c'4 c b | c2 r4 | R2. | R2. | r2 c4 | b2 c4 | R2. | c2. \bar "|."
}

>>

}
\layout {
  \context {
      \Score 
      proportionalNotationDuration =  #(ly:make-moment 1/5)
 }
 }
\midi {\tempo 4 = 60}
}
