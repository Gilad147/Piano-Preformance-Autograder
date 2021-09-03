\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
    \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \key c \major
  \time 4/4

  c2 c4 e | g2 e4 g | c,2 c2 | c1 | R1 | R1 | R1 | R1 \bar "|."
}

\new Staff = "lower" 
\relative c {
  \clef bass
  \key c \major
  \time 4/4

  R1 | R1 | R1 | R1 |c2 c4 e | g2 e4 g | c,2 c2 | c1 \bar "|."
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
