import breeze.numerics.log10
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.recommendation._

case class Artist(id:Long,name:String)

val artists=sc.textFile("/user/lhurley/music/artist_data.txt").map(_.split("\\t")).flatMap { line =>
    try {
        val artist = Artist(line(0).toLong,line(1))
        if (artist.name.isEmpty) {
                None
            } else {
                Some(artist)
            }
        } catch {
            case e: NumberFormatException => None
            case e: ArrayIndexOutOfBoundsException => None
        }
}
val rawUserArtistData = sc.textFile("/user/lhurley/music/user_artist_data.txt")
val statsForUsers=rawUserArtistData.map(_.split(' ')(0).toDouble).stats()
val statsForArtists=rawUserArtistData.map(_.split(' ')(1).toDouble).stats()

val rawArtistAlias = sc.textFile("/user/lhurley/music/artist_alias.txt")
val artistAlias = rawArtistAlias.flatMap { line =>
    val tokens = line.split('\t')
    if (tokens(0).isEmpty) {
        None
    }else{
        Some((tokens(0).toInt, tokens(1).toInt))
    }
    }.collectAsMap()
    
val bArtistAlias = sc.broadcast(artistAlias)
val trainData = rawUserArtistData.map { line =>
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    Rating(userID, finalArtistID, count)
}.cache()

val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

/*just to take a peek... */
model.userFeatures.mapValues(_.mkString(", ")).first()
/*Does the recommendation make sense? Take, for example, user 2093760. Extract the IDs of artists that this user has listened to and print their names. This means searching the input for artist IDs for this user, and then filtering the set of artists by these IDs so you can collect and print the names in order. */

val rawArtistsForUser = rawUserArtistData.map(_.split(' ')). filter { case Array(user,_,_) => user.toInt == 2093760 }
val existingProducts = rawArtistsForUser.map { case Array(_,artist,_) => artist.toInt }. collect().toSet
artists.filter { a => existingProducts.contains(a.id.toInt)  } .collect().foreach(println)