\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % Alfred’s p. 16 Good King Venceslas
  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \key c \major
  \time 4/4

  f4 f f g | f f c2 | d4 c d e | f2 f | R1 | R1 | R1 | R1 \bar "|."
}

\new Staff = "lower" 
\relative c {
  \clef bass
  \key c \major
  \time 4/4

  R1 | R1 | R1 | R1 |f4 f f g | f f c2 | d4 c d e | f2 f \bar "|."
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
