\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % trichords

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \key c \major
  \time 4/4

	
    c4 e g e | c1 | d4 f a f | d1 | b4 d f d | b1 | c4 e g e | c1 \bar "|."
}
\new Staff = "lower" 
\relative c {
  \clef bass
  \key c \major
  \time 4/4

    R1 | R1 | R1 | R1 | R1 | R1 | R1 | R1 \bar "|."
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
